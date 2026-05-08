import argparse
import json
import os
import re
import traceback

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from attn_heatmap import (
    AttentionHeatmapRunWriter,
    get_full_attention_layer_indices,
    is_qwen_attn_heatmap_model,
)
from gate_overlap_plot import clear_gate_overlap_records, save_gate_overlap_plot
from misc import (
    build_output_path,
    load_json,
    load_longbench_v2,
    load_processed_ids,
    load_prompt_templates,
    parse_domain_filter,
    select_unprocessed,
)
from models.compression.monkeypatch import replace_qwen3_5

model_map = load_json("config/model2path.json")
prompt_templates = load_prompt_templates()


def get_model_path(model_name):
    return model_map.get(model_name, model_name)

def get_max_input_len(model_maxlen, max_new_tokens):
    max_len = model_maxlen
    if max_len is None or max_len > 10 ** 8:
        max_len = 120000
    return max(1, max_len - max_new_tokens)

def truncate_prompt(prompt, tokenizer, max_input_len):
    if max_input_len is None:
        raise ValueError("max_input_len is required.")
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) > max_input_len:
        half = max_input_len // 2
        input_ids = input_ids[:half] + input_ids[-(max_input_len - half):]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    return prompt

def build_compression_config(
    compression_mode,
    compression_budget,
    gate_overlap_mode=False,
    use_linear_state=True,
    linear_state_weight=0.3,
    linear_state_required=False,
    linear_state_layer_range="all",
    linear_state_layer_reduce="mean",
    linear_state_norm="rank",
    linear_state_score_type="write_norm",
):
    method_config = {
        "budget": compression_budget,
        "window_size": 8,
        "mix_lambda": 0.07,
        "retain_ratio": 0.2,
        "retain_direction": "last",
        "first_tokens": 4,
        "record_gate_overlap": gate_overlap_mode,
        "use_linear_state": use_linear_state,
        "linear_state_weight": linear_state_weight,
        "linear_state_required": linear_state_required,
        "linear_state_layer_range": linear_state_layer_range,
        "linear_state_layer_reduce": linear_state_layer_reduce,
        "linear_state_norm": linear_state_norm,
        "linear_state_score_type": linear_state_score_type,
    }

    return {
        "method": compression_mode,
        "method_config": method_config,
        "compression": None,
        "update_kv": True,
    }


def apply_qwen3_5_compression_setup(model, tokenizer, compression_mode):
    model.config.update(
        {
            "divide_method": "step_length",
            "divide_length": 128,
            "compression_content": "think",
            "method": compression_mode,
        }
    )
    model.newline_token_ids = [
        tokenizer.encode(text, add_special_tokens=False)[-1]
        for text in ["\n", ".\n", ")\n", "\n\n", ".\n\n", ")\n\n"]
    ]
    model.after_think_token_ids = [tokenizer.encode("</think>", add_special_tokens=False)[-1]]


def load_model_and_tokenizer(
    model_name,
    attn_heatmap_mode=False,
    compression=False,
    compression_mode=None,
    compression_budget=4096,
    gate_overlap_mode=False,
    use_linear_state=True,
    linear_state_weight=0.3,
    linear_state_required=False,
    linear_state_layer_range="all",
    linear_state_layer_reduce="mean",
    linear_state_norm="rank",
    linear_state_score_type="write_norm",
):
    model_path = get_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32
    if attn_heatmap_mode:
        model_kwargs["attn_implementation"] = "eager"
    else:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if compression:
        if not compression_mode:
            raise ValueError("Please provide --compression_mode when --compression is enabled.")
        if "qwen3.5" not in model_path.lower():
            raise ValueError(
                f"Compression currently supports only qwen3.5 models, got: {model_name}"
            )

        replace_qwen3_5(
            build_compression_config(
                compression_mode,
                compression_budget,
                gate_overlap_mode=gate_overlap_mode,
                use_linear_state=use_linear_state,
                linear_state_weight=linear_state_weight,
                linear_state_required=linear_state_required,
                linear_state_layer_range=linear_state_layer_range,
                linear_state_layer_reduce=linear_state_layer_reduce,
                linear_state_norm=linear_state_norm,
                linear_state_score_type=linear_state_score_type,
            )
        )
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        apply_qwen3_5_compression_setup(model, tokenizer, compression_mode)
    elif "qwen3.5" in model_path.lower():
        from models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        from models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

        config = Qwen3_5Config.from_pretrained(model_path)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            **model_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if attn_heatmap_mode:
        text_config = getattr(model.config, "text_config", model.config)
        setattr(text_config, "_attn_implementation", "eager")
        setattr(model.config, "_attn_implementation", "eager")
    model.eval()
    return model, tokenizer

def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_inputs(prompt, tokenizer, device, enable_thinking=False):
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            inputs = input_ids if isinstance(input_ids, dict) else {"input_ids": input_ids}
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    inputs = dict(inputs)
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return {k: v.to(device) for k, v in inputs.items()}

def query_llm(
    prompt,
    model_name,
    model,
    tokenizer,
    model_maxlen,
    temperature=0.5,
    max_new_tokens=128,
    stop=None,
    enable_thinking=False,
    attn_sample_writer=None,
    prefill_label="response",
):
    max_input_len = get_max_input_len(model_maxlen, max_new_tokens)
    prompt = truncate_prompt(prompt, tokenizer, max_input_len)
    device = get_input_device(model)
    inputs = build_inputs(prompt, tokenizer, device, enable_thinking=enable_thinking)
    input_len = inputs["input_ids"].shape[-1]
    capture_record = None

    if attn_sample_writer is not None:
        capture_record = attn_sample_writer.capture_prefill(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            inputs=inputs,
            label=prefill_label,
        ).record

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "num_beams": 1,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)[0]
    return tokenizer.decode(output_ids[input_len:], skip_special_tokens=True), capture_record

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def build_prompt(template, context, item, cot=None):
    prompt = (
        template.replace('$DOC$', context.strip())
        .replace('$Q$', item['question'].strip())
        .replace('$C_A$', item['choice_A'].strip())
        .replace('$C_B$', item['choice_B'].strip())
        .replace('$C_C$', item['choice_C'].strip())
        .replace('$C_D$', item['choice_D'].strip())
    )
    if cot is not None:
        prompt = prompt.replace('$COT$', cot)
    return prompt

def build_attn_run_writer(args, out_file, model):
    if not args.attn_heatmap_mode:
        return None
    full_attention_layers = get_full_attention_layer_indices(model)
    return AttentionHeatmapRunWriter(
        root_dir=args.attn_heatmap_dir,
        model_name=args.model,
        out_file=out_file,
        full_attention_layers=full_attention_layers,
        max_prefill_tokens=args.attn_max_prefill_tokens,
    )


def validate_linear_state_layer_range(layer_range):
    if layer_range == "all":
        return
    try:
        if ":" not in layer_range:
            if int(layer_range) < 1:
                raise ValueError
            return

        if layer_range.count(":") != 1:
            raise ValueError
        start_text, end_text = layer_range.split(":", 1)
        start = int(start_text) if start_text else 1
        end = int(end_text) if end_text else None
        if start < 1 or (end is not None and end < start):
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            "--linear_state_layer_range must be 'all', 'N', ':N', or 'start:end' "
            "with 1 <= start <= end."
        ) from exc


def validate_args(args):
    if args.model_maxlen < 1:
        raise ValueError("--model_maxlen must be at least 1.")
    if args.num_samples is not None and args.num_samples < 1:
        raise ValueError("--num_samples must be at least 1 when provided.")
    if args.compression and not args.compression_mode:
        raise ValueError("--compression requires --compression_mode.")
    if args.compression and args.compression_budget < 1:
        raise ValueError("--compression_budget must be at least 1 when compression is enabled.")
    if args.gate_overlap_mode and not args.compression:
        raise ValueError("--gate_overlap_mode requires --compression.")
    if args.gate_overlap_mode and not args.compression_mode:
        raise ValueError("--gate_overlap_mode requires --compression_mode.")
    if args.gate_overlap_mode and args.n_proc != 1:
        raise ValueError("--gate_overlap_mode currently requires --n_proc 1.")
    if not 0.0 <= args.linear_state_weight <= 1.0:
        raise ValueError("--linear_state_weight must be in [0, 1].")
    if args.linear_state_layer_reduce not in {"mean", "max"}:
        raise ValueError("--linear_state_layer_reduce must be one of: mean, max.")
    if args.linear_state_norm not in {"rank", "minmax", "none"}:
        raise ValueError("--linear_state_norm must be one of: rank, minmax, none.")
    if args.linear_state_score_type not in {
        "write_norm",
        "output_norm",
        "state_similarity",
    }:
        raise ValueError(
            "--linear_state_score_type must be one of: "
            "write_norm, output_norm, state_similarity."
        )
    validate_linear_state_layer_range(args.linear_state_layer_range)
    if not args.attn_heatmap_mode:
        return
    if not is_qwen_attn_heatmap_model(args.model):
        raise ValueError("--attn_heatmap_mode currently supports only qwen3.5-* models in this repository.")
    if args.n_proc != 1:
        raise ValueError("--attn_heatmap_mode currently requires --n_proc 1.")

def get_pred(data, args, fout, out_file):
    model_name = args.model
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        attn_heatmap_mode=args.attn_heatmap_mode,
        compression=args.compression,
        compression_mode=args.compression_mode,
        compression_budget=args.compression_budget,
        gate_overlap_mode=args.gate_overlap_mode,
        use_linear_state=args.use_linear_state,
        linear_state_weight=args.linear_state_weight,
        linear_state_required=args.linear_state_required,
        linear_state_layer_range=args.linear_state_layer_range,
        linear_state_layer_reduce=args.linear_state_layer_reduce,
        linear_state_norm=args.linear_state_norm,
        linear_state_score_type=args.linear_state_score_type,
    )
    attn_run_writer = build_attn_run_writer(args, out_file, model)
    for sample_index, item in enumerate(tqdm(data)):
        item = dict(item)
        attn_sample_writer = attn_run_writer.new_sample(item) if attn_run_writer is not None else None
        if args.gate_overlap_mode:
            clear_gate_overlap_records(model)
        try:
            context = item['context']
            if args.rag > 0:
                template = prompt_templates["rag"]
                retrieved = item["retrieved_context"][:args.rag]
                retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
                context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
            elif args.no_context:
                template = prompt_templates["no_context"]
            elif args.cot:
                template = prompt_templates["cot"]
            else:
                template = prompt_templates["zero_shot"]
            prompt = build_prompt(template, context, item)
            if args.cot:
                output, _ = query_llm(
                    prompt,
                    model_name,
                    model,
                    tokenizer,
                    args.model_maxlen,
                    temperature=0.1,
                    max_new_tokens=1024,
                    enable_thinking=args.cot,
                    attn_sample_writer=attn_sample_writer,
                    prefill_label="cot_reasoning",
                )
            else:
                output, _ = query_llm(
                    prompt,
                    model_name,
                    model,
                    tokenizer,
                    args.model_maxlen,
                    temperature=0.1,
                    max_new_tokens=128,
                    enable_thinking=args.cot,
                    attn_sample_writer=attn_sample_writer,
                    prefill_label="response",
                )
            if output == '':
                continue
            if args.cot: # extract answer
                response = output.strip()
                item['response_cot'] = response
                prompt = build_prompt(prompt_templates["cot_answer"], context, item, cot=response)
                output, _ = query_llm(
                    prompt,
                    model_name,
                    model,
                    tokenizer,
                    args.model_maxlen,
                    temperature=0.1,
                    max_new_tokens=128,
                    enable_thinking=args.cot,
                    attn_sample_writer=attn_sample_writer,
                    prefill_label="cot_answer_extraction",
                )
                if output == '':
                    continue
            response = output.strip()
            item['response'] = response
            item['pred'] = extract_answer(response)
            item['judge'] = item['pred'] == item['answer']
            item['context'] = context[:1000]
            if attn_sample_writer is not None:
                item["attn_capture_status"] = attn_sample_writer.build_capture_status()
                item["attn_artifact"] = os.path.relpath(attn_sample_writer.sample_dir, start=args.attn_heatmap_dir)
            gate_overlap_path = save_gate_overlap_plot(
                model,
                args,
                out_file,
                item,
                sample_index,
            )
            if gate_overlap_path is not None:
                item["gate_overlap_artifact"] = os.path.relpath(
                    gate_overlap_path,
                    start=args.gate_overlap_dir,
                )
            if attn_sample_writer is not None:
                attn_sample_writer.finalize(item)
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            fout.flush()
        except Exception as exc:
            sample_id = item.get("_id", "unknown")
            print(f"Skipping sample {sample_id} due to error: {exc}")
            traceback.print_exc()
            if attn_sample_writer is not None:
                item["error"] = str(exc)
                item["attn_capture_status"] = attn_sample_writer.build_capture_status()
                item["attn_artifact"] = os.path.relpath(attn_sample_writer.sample_dir, start=args.attn_heatmap_dir)
                attn_sample_writer.finalize(item)
            continue
        finally:
            if args.gate_overlap_mode:
                clear_gate_overlap_records(model)

def main(args):
    print(args)
    validate_args(args)
    domains = parse_domain_filter(args.domain)
    out_file = build_output_path(args, domains)
    print(f"Writing results to {out_file}")

    data_all = load_longbench_v2(domains)
    processed_ids = load_processed_ids(out_file)
    data = select_unprocessed(data_all, processed_ids)
    if args.num_samples is not None:
        data = data[:args.num_samples]
        print(f"Limited this run to {len(data)} example(s).")

    if len(data) == 0:
        print("No new examples to process.")
        return

    with open(out_file, 'a', encoding='utf-8') as fout:
        if args.n_proc == 1:
            get_pred(data, args, fout, out_file)
        else:
            print("Warning: each process will load its own transformers model copy.")
            data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
            processes = []
            for rank in range(args.n_proc):
                p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout, out_file))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="output_dir/results_longbench")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--model_maxlen", type=int, default=120000, help="Model context length used for prompt truncation.")
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--domain", "-d", action='append', default=None, help="Only run examples from the given domain. Repeat this option or use comma-separated names for multiple domains.")
    parser.add_argument("--num_samples", "--max_samples", type=int, default=None, help="Only run the first N unprocessed examples.")
    parser.add_argument("--n_proc", "-n", type=int, default=1)
    parser.add_argument("--compression", action="store_true")
    parser.add_argument("--compression_mode", type=str, default=None)
    parser.add_argument("--compression_budget", type=int, default=4096)
    parser.add_argument("--attn_heatmap_mode", action="store_true")
    parser.add_argument("--attn_heatmap_dir", type=str, default="output_dir/results_longbench/attn_heatmaps")
    parser.add_argument("--attn_max_prefill_tokens", type=int, default=None, help="Skip attention heatmap capture when the prefill token count exceeds this cap.")
    parser.add_argument("--gate_overlap_mode", "--gate_method_overlap_mode", action="store_true", help="Record gate topk overlap with the active compression method and save one layer/head heatmap per sample.")
    parser.add_argument("--use_linear_state", action=argparse.BooleanOptionalAction, default=True, help="Enable GatedDeltaNet linear-state score enhancement in compression method config.")
    parser.add_argument("--linear_state_weight", type=float, default=0.3, help="Weight of linear-state score when fusing with gate score.")
    parser.add_argument("--linear_state_required", action="store_true", help="Raise an error if linear-state scores are unavailable.")
    parser.add_argument("--linear_state_layer_range", type=str, default="all", help="Preceding linear-attention layers to aggregate: all, N for only the Nth nearest layer, :N for nearest N layers, or 1-indexed start:end from nearest to farthest.")
    parser.add_argument("--linear_state_layer_reduce", type=str, default="mean", choices=["mean", "max"], help="How to reduce scores from preceding linear-attention layers.")
    parser.add_argument("--linear_state_norm", type=str, default="rank", choices=["rank", "minmax", "none"], help="Normalization used before gate/linear-state score fusion.")
    parser.add_argument("--linear_state_score_type", type=str, default="write_norm", choices=["write_norm", "output_norm", "state_similarity"], help="Token importance definition for linear-state enhancement.")
    parser.add_argument("--gate_overlap_dir", type=str, default="output_dir/results_longbench/gate_overlaps")
    parser.add_argument("--gate_overlap_interpolation", type=str, default="bicubic", help="Matplotlib interpolation mode for the gate/method overlap heatmap.")
    args = parser.parse_args()
    main(args)
