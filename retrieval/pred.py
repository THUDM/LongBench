import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModel, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import argparse
# DEBUG
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="chatglm2-6b")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--data", type=str, default="B500")
    return parser.parse_args(args)

# This is the customized building prompt for chat models, here is an example for ChatGLM2
def build_chat(tokenizer, prompt, model_name):
    if "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()        
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, args):
    preds = [{}] * len(data)
    if os.path.exists(f"{args.model}_pred_{args.data}_{args.top_k}/{dataset}.jsonl"):
        with open(f"{args.model}_pred_{args.data}_{args.top_k}/{dataset}.jsonl", "r", encoding="utf-8") as f:
            for index, item in enumerate(f):
                preds[index] = json.loads(item)
    for index, json_obj in enumerate(tqdm(data, desc=f"{dataset}")):
        if preds[index] != {}:
            continue
        if args.top_k != 0:
            json_obj['context'] = "".join(json_obj['retrieved'][:args.top_k])
        prompt = prompt_format.format(**json_obj)
        prompt = build_chat(tokenizer, prompt, model_name)
        if "chatgpt" in model_name:
            output = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                 messages=[{"role": "user", "content": prompt}], max_tokens=max_gen,
                 temperature=1.0)
            pred = output['choices'][0]['message']['content']
            context_length = output['usage']['prompt_tokens']
        else:
            # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompt on these tasks
                prompt = build_chat(tokenizer, prompt, model_name)
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            
            context_length = input.input_ids.shape[-1]
            if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds[index] = {"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"],
                        "context_length": context_length}
        with open(f"{args.model}_pred_{args.data}_{args.top_k}/{dataset}.jsonl", "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')
    return preds

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def load_model_and_tokenizer(model2path, model_name, device):
    if "chatgpt" in model_name:
        return model_name, model_name
    else:
        if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model2path[model_name], trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model2path[model_name], trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
        elif "llama2" in model_name:
            tokenizer = LlamaTokenizer.from_pretrained(model2path[model_name])
            model = LlamaForCausalLM.from_pretrained(model2path[model_name], torch_dtype=torch.bfloat16).to(device)
        elif "longchat" in model_name or "vicuna" in model_name:
            from fastchat.model import load_model
            model, _ = load_model(
                model2path[model_name],
                device='cpu',
                num_gpus=0,
                load_8bit=False,
                cpu_offloading=False,
                debug=False,
            )
            model = model.to(device)
            model = model.bfloat16()
            tokenizer = AutoTokenizer.from_pretrained(model2path[model_name], trust_remote_code=True, use_fast=False)
        model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    args = parse_args()
    model_name = args.model
    if "chatgpt" in model_name:
        import openai
        # openai.api_base=""
        openai.api_key = "YOUR_KEY"
    # Retrieval is fit for these datasets
    datasets = ["multifieldqa_en", "qasper", "2wikimqa", "dureader", \
                "hotpotqa", "narrativeqa", "musique", "multifieldqa_zh"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load configs
    model2path = json.load(open("../config/model2path.json", "r"))
    model2maxlen = json.load(open("../config/model2maxlen.json", "r"))
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("../config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("../config/dataset2maxlen.json", "r"))
    # define your model
    model, tokenizer = load_model_and_tokenizer(model2path, model_name, device)
    max_length = model2maxlen[model_name]
    # predict on each dataset
    os.makedirs(f"{args.model}_pred_{args.data}_{args.top_k}", exist_ok=True)
    for dataset in datasets:
        data = load_dataset(f'../LongBench/{args.data}/LongBench.py', dataset, split='test',
                            download_mode='force_redownload') # force to load from dir
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, args)