import os
import json
import argparse
import numpy as np

def string_match_all(preds, refs):
    """
    evaluation metric for RULER
    preds: List[str]
    refs: List[List[str]]
    """
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    
    # Read JSON files from results_dir, excluding metrics.json
    niah_single_1_dir = os.path.join(args.results_dir, "niah_single_1")
    # Get all filenames except metrics.json
    json_files = [f for f in os.listdir(niah_single_1_dir) if f.endswith('.json') and f != "metrics.json"]
    
    # Extract method names from filenames
    methods = [[os.path.splitext(f)[0]] for f in json_files]
    
    # Build results list
    results_list = [["dataset"]]
    results_list.extend(methods)  # Append methods

    dataset_list = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
            "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt"]

    results_list = [
        ["dataset"],
        ["FullKV"],
        ["random"],
        ["SnapKV"],
        ["StreamingLLM"],
        ["H2O"],
        ["PyramidKV"],
        ["L2Norm"],
        ["ReST-KV"],
    ]

    for dataset in dataset_list:

        results_list[0].append(dataset)

        for idx, method in enumerate(methods):
            try:
                args.method = method[0]
                args.dataset = dataset
                args.eval_file = os.path.join(args.results_dir,dataset,f"{method[0]}.json")

                scores = dict()
                predictions, answers, lengths = [], [], []
                # dataset = filename.split('.')[0]
                with open(args.eval_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            predictions.append(data["pred"])
                            answers.append(data["answers"])
                            if "length" in data:
                                lengths.append(data["length"])
                        except:
                            print("error")

                score = string_match_all(predictions, answers)
                scores[args.dataset] = score
                results_list[idx+1].append(score)

                output_dir = os.path.dirname(args.eval_file)
                with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                    json.dump(scores, f, ensure_ascii=False, indent=4)
                print(f"dataset {args.dataset} method {args.method} scores {scores}")

            except:
                results_list[idx+1].append(-1)
                print(f"dataset {args.dataset} method {args.method} scores {None}")

    import csv
    with open(os.path.join(args.results_dir,f"results.csv"), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(results_list)