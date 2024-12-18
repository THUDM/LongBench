import os
import json
import pandas as pd
import argparse
import re
from tqdm import tqdm

import sys
sys.path.append('..')
from splitter import split_long_sentence, regex
import concurrent.futures

# DEBUG
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, default='../source/docqa_only')
parser.add_argument("--chunk_size", type=int, default=200)
parser.add_argument("--output_folder", type=str, default='../datasets/C200_t/split')
args = parser.parse_args()



def process_jsonl_file(input_file, output_folder, chunk_size=100, filename='Unknown'):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
        # for idx, line in enumerate(lines):
        loop = tqdm(lines, desc=filename)
        for line in loop:
            data = json.loads(line)
            context = data.get('context', '')
            chunks = split_long_sentence(context, regex, chunk_size, filename)
            output_folder_name = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0])
            if not os.path.exists(output_folder_name):
                os.makedirs(output_folder_name)
            output_data = []
            for i, chunk in enumerate(chunks):
                output_datum = {
                    'id': data['_id'] + '_' + str(i),
                    'text': chunk.strip(),
                    'title': ''
                }
                output_data.append(output_datum)
            output_data = pd.DataFrame(output_data, index=range(len(output_data)))
            output_tsv_file = os.path.join(output_folder_name, data['_id'] + '.tsv')
            output_data.to_csv(output_tsv_file, sep='\t', index=False)

            output_jsonl_file = os.path.join(output_folder_name, data['_id'] + '.jsonl')
            output_data = {
                'id': data['_id'],
                # 'lang': 'zh' if "_zh" in input_file else 'en',
                'lang' : 'zh' if 'zh' in data.get('context', '') else 'en',
                'question': data.get('input', ''),
                'answers': []
            }
            with open(output_jsonl_file, 'w', encoding='utf-8') as f_out:
                f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')

def process_all_jsonl_files(input_folder, output_folder, chunk_size=1700):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    loop = tqdm(os.listdir(input_folder))
    allowed_files = ["multifieldqa_en.jsonl", "qasper.jsonl", "2wikimqa.jsonl", "dureader.jsonl", "hotpotqa.jsonl", "narrativeqa.jsonl", "musique.jsonl", "multifieldqa_zh.jsonl"]
    for filename in loop:
        if filename.endswith('.jsonl') and filename in allowed_files:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                input_file = os.path.join(input_folder, filename)
                loop.set_description(f"totalFile")
                # process_jsonl_file(input_file, output_folder, chunk_size, filename)
                executor.submit(process_jsonl_file, input_file, output_folder, chunk_size, filename)
                # print("split {} done!".format(filename))

process_all_jsonl_files(args.input_folder, args.output_folder, chunk_size=args.chunk_size)
