import os
import json
import random
import argparse
import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

from models.compression.monkeypatch import replace_qwen3_5


context_length_list = [8192]
# context_length_list = [4096, 8192, 16384]

datasets = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
            "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt"]

dataset2maxlen = {
    "niah_single_1": 64,
    "niah_single_2": 64,
    "niah_single_3": 64,
    "niah_multikey_1": 64,
    "niah_multikey_2": 64,
    "niah_multikey_3": 64,
    "niah_multiquery": 64,
    "niah_multivalue": 64,
    "cwe": 64,
    "fwe": 64,
    "vt": 64
}


model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 7950,
    "llama-3-70": 7950,
    "llama-3.1": 127000,
    "mistral": 31500,
    "qwen2.5": 127000,
    "gemma": 7950,
    "qwen3.5": 200000,
}



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def get_max_input_len(model_maxlen, max_new_tokens):
    max_len = model_maxlen
    if max_len is None or max_len > 10 ** 8:
        max_len = 120000
    return max(1, max_len - max_new_tokens)


def truncate_prompt(prompt, tokenizer, max_input_len):
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) > max_input_len:
        half = max_input_len // 2
        input_ids = input_ids[:half] + input_ids[-(max_input_len - half):]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    return prompt


def build_compression_config(
    compression_mode,
    compression_budget,
    use_linear_state=False,
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
        "use_linear_state": use_linear_state,
    }
    if use_linear_state:
        method_config.update(
            {
                "linear_state_weight": linear_state_weight,
                "linear_state_required": linear_state_required,
                "linear_state_layer_range": linear_state_layer_range,
                "linear_state_layer_reduce": linear_state_layer_reduce,
                "linear_state_norm": linear_state_norm,
                "linear_state_score_type": linear_state_score_type,
            }
        )

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


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "attn_implementation": args.attn_implementation,
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model_path_lower = args.model_path.lower()
    if args.compression:
        if not args.compression_mode:
            raise ValueError("Please provide --compression_mode when --compression is enabled.")
        if "qwen3.5" not in model_path_lower:
            raise ValueError(
                f"Compression currently supports only qwen3.5 models, got: {args.model_path}"
            )

        replace_qwen3_5(
            build_compression_config(
                args.compression_mode,
                args.compression_budget,
                use_linear_state=args.use_linear_state,
                linear_state_weight=args.linear_state_weight,
                linear_state_required=args.linear_state_required,
                linear_state_layer_range=args.linear_state_layer_range,
                linear_state_layer_reduce=args.linear_state_layer_reduce,
                linear_state_norm=args.linear_state_norm,
                linear_state_score_type=args.linear_state_score_type,
            )
        )
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
        apply_qwen3_5_compression_setup(model, tokenizer, args.compression_mode)
    elif "qwen3.5" in model_path_lower:
        from models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        from models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

        config = Qwen3_5Config.from_pretrained(args.model_path)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model_path,
            config=config,
            **model_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    model.eval()
    return model, tokenizer


def validate_args(args):
    if not args.model_path:
        raise ValueError("--model_path is required.")
    if args.model_maxlen < 1:
        raise ValueError("--model_maxlen must be at least 1.")
    if args.compression and not args.compression_mode:
        raise ValueError("--compression requires --compression_mode.")
    if args.compression and args.compression_budget < 1:
        raise ValueError("--compression_budget must be at least 1 when compression is enabled.")
    if args.use_linear_state:
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


def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_inputs(prompts, tokenizer, device, enable_thinking=False):
    if isinstance(prompts, str):
        prompts = [prompts]

    if getattr(tokenizer, "chat_template", None):
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    padding=True,
                    return_tensors="pt",
                    enable_thinking=enable_thinking,
                )
                inputs = input_ids if isinstance(input_ids, dict) else {"input_ids": input_ids}
            except TypeError:
                rendered_prompts = []
                for message in messages:
                    try:
                        rendered_prompt = tokenizer.apply_chat_template(
                            message,
                            add_generation_prompt=True,
                            tokenize=False,
                            enable_thinking=enable_thinking,
                        )
                    except TypeError:
                        rendered_prompt = tokenizer.apply_chat_template(
                            message,
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                    rendered_prompts.append(rendered_prompt)
                inputs = tokenizer(
                    rendered_prompts,
                    padding="longest",
                    return_tensors="pt",
                    add_special_tokens=False,
                )
    else:
        inputs = tokenizer(
            prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=True,
        )

    inputs = dict(inputs)
    if "attention_mask" not in inputs:
        if tokenizer.pad_token_id is not None:
            inputs["attention_mask"] = inputs["input_ids"].ne(tokenizer.pad_token_id).long()
        else:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

# def build_prompt(prompt, dataset):

#     SYSTEM_PROMPT = model2prompt[dataset]

#     prompt = f"<<SYS>>\n {SYSTEM_PROMPT} \n<</SYS>>\n\n{prompt}"
#     return prompt

def main(args):


    print("Loading data...")

    test_data = []
    prompt_list = []
    input_list = []
    outputs_list: List[List[str]] = [] # List of List
    length_list = []
    index_list = []

    input_max_len = 0
    output_max_len = dataset2maxlen[args.dataset]
    max_input_len = get_max_input_len(args.model_maxlen, output_max_len)

    with open(args.data_file) as fp:
        for line in fp:

            example = json.loads(line)
            length = example["length"]
            if length > input_max_len:
                input_max_len = length

            prompt = example["input"]
            example["prompt"] = prompt

            test_data.append(example)

    print(f"Max Length is {input_max_len}")

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]

    for example in test_data:
        prompt_list.append(example["prompt"])
        input_list.append(example["input"])
        outputs_list.append(example["outputs"])
        length_list.append(example["length"])
        index_list.append(example["index"])

    print("Finish loading model and tokenizer")
    model_name = args.model_path.rstrip("/").split("/")[-1]
    run_name = args.compression_mode if args.compression else "FullKV"
    budget_name = str(args.compression_budget) if args.compression else "full"

    os.makedirs(os.path.join(args.save_dir, f"{model_name}_{budget_name}", str(args.context_length), args.dataset), exist_ok=True)
    fout = open(os.path.join(args.save_dir, f"{model_name}_{budget_name}", str(args.context_length), args.dataset, f"{run_name}_{args.add_file_name}.json"), "w")

    for i in tqdm(range(0, len(prompt_list), args.eval_batch_size)):

        batch_prompts = prompt_list[i:i+args.eval_batch_size]
        batch_inputs = input_list[i:i+args.eval_batch_size]
        batch_answers = outputs_list[i:i+args.eval_batch_size]
        batch_lengths = length_list[i:i+args.eval_batch_size]

        batch_prompts = [
            truncate_prompt(prompt, tokenizer, max_input_len)
            for prompt in batch_prompts
        ]
        tokenized_prompts = build_inputs(
            batch_prompts,
            tokenizer,
            get_input_device(model),
            enable_thinking=args.enable_thinking,
        )
        batch_input_ids = tokenized_prompts["input_ids"]

        context_length = batch_input_ids.shape[-1]
        if args.quant_method == None:
            output = model.generate(
                **tokenized_prompts,
                output_attentions = args.output_attentions,
                max_new_tokens=output_max_len,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id],
                pad_token_id=tokenizer.pad_token_id
            )
        else:
            output = model.generate(
                **tokenized_prompts,
                output_attentions = args.output_attentions,
                max_new_tokens=output_max_len,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id],
                cache_implementation="quantized",
                cache_config={"nbits": args.nbits, "backend": "HQQ","device":"cuda","residual_length":output_max_len,"axis_key":1,"q_group_size":64},
            )

        batch_outputs = tokenizer.batch_decode(output[:, context_length:], skip_special_tokens=True)
        batch_generations = batch_outputs

        torch.cuda.empty_cache()

        for j in range(len(batch_prompts)):

            example = {}
            example["prompt"] = batch_prompts[j]
            example["input"] = batch_inputs[j]
            example["answers"] = batch_answers[j]
            example["pred"] = batch_generations[j]
            example["length"] = batch_lengths[j]

            fout.write(json.dumps(example) + "\n")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--add_file_name", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_maxlen", type=int, default=120000, help="Model context length used for prompt truncation.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    parser.add_argument("--enable_thinking", action="store_true", help="Pass enable_thinking=True to chat templates that support it.")

    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")

    parser.add_argument("--max_new_tokens", type=int, default=None, help="")

    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")

    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--quant_method",type=str,default=None,choices=["kivi","kvquant"])
    parser.add_argument("--nbits", type=int, default=8, help="")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--compression", action="store_true")
    parser.add_argument("--compression_mode", type=str, default=None)
    parser.add_argument("--compression_budget", type=int, default=4096)
    parser.add_argument("--use_linear_state", action="store_true", help="Enable GatedDeltaNet linear-state score enhancement in compression method config.")
    parser.add_argument("--linear_state_weight", type=float, default=0.3, help="Weight of linear-state score when fusing with gate score.")
    parser.add_argument("--linear_state_required", action="store_true", help="Raise an error if linear-state scores are unavailable.")
    parser.add_argument("--linear_state_layer_range", type=str, default="all", help="Preceding linear-attention layers to aggregate: all, N for only the Nth nearest layer, :N for nearest N layers, or 1-indexed start:end from nearest to farthest.")
    parser.add_argument("--linear_state_layer_reduce", type=str, default="mean", help="How to reduce scores from preceding linear-attention layers.")
    parser.add_argument("--linear_state_norm", type=str, default="rank", help="Normalization used before gate/linear-state score fusion.")
    parser.add_argument(
        "--linear_state_score_type",
        type=str,
        default="write_norm",
        help="Token importance definition for linear-state enhancement.",
    )

    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )

    args = parser.parse_args()

    validate_args(args)
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)

    for context_length in context_length_list:
        for idx, dataset in enumerate(datasets):

            run_name = args.compression_mode if args.compression else "FullKV"
            print(f"Working on context length {context_length}, method: {run_name}, dataset: {dataset} - {idx}/{len(datasets)}")
            args.context_length = context_length
            args.dataset = dataset
            args.data_file = f"data/RULER/{context_length}/{args.dataset}.jsonl"

            main(args)
