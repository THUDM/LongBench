import csv
import json
import os
from collections import defaultdict

RESULTS_DIR = "results"
DOMAIN_ORDER = [
    "Long-dialogue History Understanding",
    "Single-Document QA",
    "Multi-Document QA",
    "Long Structured Data Understanding",
]


def pct(score, total):
    return round(100 * score / total, 1) if total else 0.0


files = [
    file for file in os.listdir(RESULTS_DIR)
    if os.path.isfile(os.path.join(RESULTS_DIR, file))
]
output = [[
    "Model",
    "Overall",
    "Easy",
    "Hard",
    "Short",
    "Medium",
    "Long",
    *DOMAIN_ORDER,
]]
compensated = False

for file in files:
    filename = os.path.join(RESULTS_DIR, file)
    try:
        pred_data = json.load(open(filename, encoding='utf-8'))
    except Exception as e:
        pred_data = [json.loads(line) for line in open(filename, encoding='utf-8')]
    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    domain_total = defaultdict(int)
    domain_acc = defaultdict(float)
    for pred in pred_data:
        acc = int(pred['judge'])
        if compensated and pred["pred"] == None:
            acc = 0.25
        if pred["difficulty"] == "easy":
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc

        if pred['length'] == "short":
            short += 1
            short_acc += acc
        elif pred['length'] == "medium":
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc

        domain = pred.get("domain")
        if domain:
            domain_total[domain] += 1
            domain_acc[domain] += acc

    name = '.'.join(file.split('.')[:-1])
    row = [
        name,
        str(pct(easy_acc + hard_acc, len(pred_data))),
        str(pct(easy_acc, easy)),
        str(pct(hard_acc, hard)),
        str(pct(short_acc, short)),
        str(pct(medium_acc, medium)),
        str(pct(long_acc, long)),
    ]
    row.extend(str(pct(domain_acc[domain], domain_total[domain])) for domain in DOMAIN_ORDER)
    output.append(row)

with open('result.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(output)
