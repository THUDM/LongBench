import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# This is the customized building prompt for chat models, here is an example for ChatGLM2
def build_chat(tokenizer, prompt):
    return tokenizer.build_prompt(prompt)

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["lcc", "repobench-p", "trec", "nq", "triviaqa", "lsht"]: # chat models are better off without build prompt on these tasks
            prompt = build_chat(tokenizer, prompt)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        output = model.generate(
            **input,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"]})
    return preds


if __name__ == '__main__':
    datasets = ["hotpotqa", "2wikimqa", "musique", "dureader", "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "gov_report", \
        "qmsum", "vcsum", "trec", "nq", "triviaqa", "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model (ChatGLM2-6B, for instance)
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    # define max_length
    max_length = 31500
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    for dataset in datasets:
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device)
        with open(f"pred/{dataset}.jsonl", "w") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')