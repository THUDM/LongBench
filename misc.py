import json
import os
import re
from datetime import datetime

from datasets import load_dataset


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_prompt_templates(prompt_dir="prompts"):
    return {
        "rag": _read_text(os.path.join(prompt_dir, "0shot_rag.txt")),
        "no_context": _read_text(os.path.join(prompt_dir, "0shot_no_context.txt")),
        "zero_shot": _read_text(os.path.join(prompt_dir, "0shot.txt")),
        "cot": _read_text(os.path.join(prompt_dir, "0shot_cot.txt")),
        "cot_answer": _read_text(os.path.join(prompt_dir, "0shot_cot_ans.txt")),
    }


def parse_domain_filter(domain_args):
    if not domain_args:
        return None
    domains = []
    for domain_arg in domain_args:
        for domain in domain_arg.split(","):
            domain = domain.strip()
            if domain and domain not in domains:
                domains.append(domain)
    return domains or None


def get_domain_suffix(domains):
    if not domains:
        return ""
    safe_domains = []
    for domain in domains:
        safe_domain = re.sub(r"[^A-Za-z0-9._-]+", "_", domain).strip("_")
        if safe_domain:
            safe_domains.append(safe_domain)
    return "_domain_" + "_".join(safe_domains) if safe_domains else ""


def filter_by_domain(data, domains):
    if not domains:
        return data
    domain_set = {_normalize_domain(domain) for domain in domains}
    available_domains = sorted({item["domain"].strip() for item in data})
    available_domain_set = {_normalize_domain(domain) for domain in available_domains}
    missing_domains = [
        domain for domain in domains if _normalize_domain(domain) not in available_domain_set
    ]
    if missing_domains:
        raise ValueError(
            f"Domain(s) not found: {', '.join(missing_domains)}. "
            f"Available domains: {', '.join(available_domains)}"
        )
    filtered_data = [
        item for item in data if _normalize_domain(item["domain"]) in domain_set
    ]
    print(
        f"Selected {len(filtered_data)} / {len(data)} examples for domain(s): "
        f"{', '.join(domains)}"
    )
    return filtered_data


def build_output_path(args, domains):
    os.makedirs(args.save_dir, exist_ok=True)
    output_prefix = args.model.split("/")[-1] + get_domain_suffix(domains)
    if getattr(args, "compression", False):
        compression_mode = getattr(args, "compression_mode", None) or "compressed"
        compression_budget = getattr(args, "compression_budget", None)
        if compression_budget is not None:
            output_prefix = f"{output_prefix}_{compression_mode}_budget_{compression_budget}"
        else:
            output_prefix = f"{output_prefix}_{compression_mode}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.rag > 0:
        filename = f"{output_prefix}_rag_{args.rag}_{timestamp}.jsonl"
    elif args.no_context:
        filename = f"{output_prefix}_no_context_{timestamp}.jsonl"
    elif args.cot:
        filename = f"{output_prefix}_cot_{timestamp}.jsonl"
    else:
        filename = f"{output_prefix}_{timestamp}.jsonl"
    return os.path.join(args.save_dir, filename)


def load_longbench_v2(domains=None):
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    data = [_build_example(item) for item in dataset]
    return filter_by_domain(data, domains)


def load_processed_ids(out_file):
    if not os.path.exists(out_file):
        return set()
    with open(out_file, encoding="utf-8") as f:
        return {json.loads(line)["_id"] for line in f}


def select_unprocessed(data, processed_ids):
    return [item for item in data if item["_id"] not in processed_ids]


def _read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _build_example(item):
    example = {
        "_id": item["_id"],
        "domain": item["domain"],
        "sub_domain": item["sub_domain"],
        "difficulty": item["difficulty"],
        "length": item["length"],
        "question": item["question"],
        "choice_A": item["choice_A"],
        "choice_B": item["choice_B"],
        "choice_C": item["choice_C"],
        "choice_D": item["choice_D"],
        "answer": item["answer"],
        "context": item["context"],
    }
    if "retrieved_context" in item:
        example["retrieved_context"] = item["retrieved_context"]
    return example


def _normalize_domain(domain):
    return domain.strip().lower()
