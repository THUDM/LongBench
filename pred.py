import os
from datasets import load_dataset
import torch
import json
from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import (DynamicCache,
                          SinkCache,
                          SlidingWindowCache,
                          QuantoQuantizedCache,
                          )
import gc
from vllm import LLM, SamplingParams

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,)
    parser.add_argument('--max_seq_len', type=int, default=7500,)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--kv_method', type=str, default='full',)
    parser.add_argument('--prefill_kv', action='store_true', help="prefill kv cache before generate function")
    parser.add_argument('--backend', type=str, default='hf',choices=["hf","vllm"])
    parser.add_argument('--load_in_8bit', action='store_true')
    return parser.parse_args(args)

def prefill_kv_cache(past_key_values,model,input,max_gen_len):
    input_ids = input['input_ids']
    for idx in range(input_ids.shape[1]-1):
        with torch.no_grad():
            outputs = model(
                input_ids[:, idx : idx + 1].to(device),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
    return past_key_values

def get_pred(data, max_length, max_gen, prompt_format, dataset, device, model, tokenizer, model_config, out_path, args=None):
    # device = torch.device(f'cuda:{rank}')
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if args.kv_method == 'full':
            cache = DynamicCache()
        elif args.kv_method == 'sink':
            cache = SinkCache(window_length=2048, num_sink_tokens=8)
        elif args.kv_method == 'sliding':
            setattr(model_config, 'sliding_window', 2048)
            cache = SlidingWindowCache(config=model_config,
                                       max_batch_size=1,
                                       max_cache_len=2048,
                                       device='cuda')
        if args.prefill_kv:
            cache = prefill_kv_cache(cache,model,input,max_gen)
        if args.backend == "hf":#dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            with torch.no_grad():
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                    pad_token_id = tokenizer.eos_token_id,
                    past_key_values=cache,
                    # return_dict_in_generate=True,
                )[0]
        else:
            with torch.no_grad():
                output = model.generate(
                    prompt,
                    SamplingParams(
                        temperature=1.0,
                        max_tokens=max_gen,
                    )
                    # max_new_tokens=max_gen,
                    # num_beams=1,
                    # do_sample=False,
                    # temperature=1.0,
                    # pad_token_id = tokenizer.eos_token_id,
                    # past_key_values=cache,
                    # return_dict_in_generate=True,
                )
        if args.backend == "hf":
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            # print(output[0].outputs[0].text)
        else:
            pred = output[0].outputs[0].text
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
        del input, output, pred
        gc.collect()
        torch.cuda.empty_cache()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path,load_in_8bit,backend):
    model_config = AutoConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    if backend == "vllm":
        model = LLM(path)
    elif backend == "hf":
        if load_in_8bit:
            model = AutoModelForCausalLM.from_pretrained(path, load_in_8bit=True,device_map='auto',attn_implementation='flash_attention_2')
        else:
            model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,device_map='auto',attn_implementation='flash_attention_2')#.to(device)
            model = model.eval()
    return model, tokenizer, model_config

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer, model_config = load_model_and_tokenizer(model_name, args.load_in_8bit,args.backend)
    max_length = args.max_seq_len
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",\
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    output_dir_name = args.model.split('/')[-1]+ "_" + str(args.max_seq_len) + "_kv_cache_" + args.kv_method + "_backend_" + args.backend
    if args.prefill_kv:
        output_dir_name = output_dir_name + '_prefill_kv'
    if args.load_in_8bit:
        output_dir_name = output_dir_name + '_8bit'
    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{output_dir_name}"):
                os.makedirs(f"pred_e/{output_dir_name}")
            out_path = f"pred_e/{output_dir_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{output_dir_name}"):
                os.makedirs(f"pred/{output_dir_name}")
            out_path = f"pred/{output_dir_name}/{dataset}.jsonl"
        if os.path.isfile(out_path) and os.access(out_path, os.R_OK):
            print(f'Skipping {dataset}. Predictions already generated')
            continue
        print(f'Getting prediction on {dataset}.')
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        processes = []
        get_pred(data_all, max_length, max_gen, prompt_format, dataset, device, model, tokenizer, model_config, out_path, args)
        