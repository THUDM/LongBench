import os
import json
import argparse
from tqdm import tqdm
import sys
sys.path.append('..')
from splitter import get_word_len

# os.chdir(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, default='mcontriever_output',
                        help='Path to the input folder containing jsonl files.')
parser.add_argument('--output_file', type=str, default='CONTENT.jsonl',
                        help='Output jsonl file name.')
parser.add_argument('--input_dataFile', type=str, default='inputData.jsonl',
                        help='Input datum jsonl file name.')
parser.add_argument('--output_dataFile', type=str, default='DATA.jsonl',
                        help='Output datum jsonl file name.')
args = parser.parse_args()


def merge_text(jsonl_file, maxLen=1500):
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    context_list = data_list['ctxs']
    merged_text = ''
    retrieved = []
    for item in context_list:
        if get_word_len(merged_text) < maxLen:
            merged_text += item['text'] + '\n\n'
        retrieved += [item['text']]
    output_data = {
        'context': merged_text,
        'id': data_list['id'],
        'retrieved': retrieved
    }

    return output_data

def process_all_jsonl_files(args):
    input_folder = args.input_folder
    output_file = args.output_file
    # data_name = os.path.basename(os.path.normpath(input_folder))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_data_list = []
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # print("input_folder", input_folder)
        loop = tqdm(os.listdir(input_folder), desc="merge")
        for filename in loop:
            if filename.endswith('.jsonl'):
                jsonl_file = os.path.join(input_folder, filename)
                output_data = merge_text(jsonl_file)
                output_data_list += [output_data]
                f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
    os.makedirs(os.path.dirname(args.output_dataFile), exist_ok=True)
    with open(args.input_dataFile, 'r', encoding='utf-8') as in_data:
        with open(args.output_dataFile, 'w', encoding='utf-8') as out_data:
            for line in in_data:
                data_l = json.loads(line)
                for modified_data in output_data_list:
                    if data_l['_id'] == modified_data['id']:
                        data_l['context'] = modified_data['context']
                        data_l['length'] = get_word_len(data_l['context'])
                        data_l['retrieved'] = modified_data['retrieved']
                        break
                out_data.write(json.dumps(data_l, ensure_ascii=False) + '\n')

process_all_jsonl_files(args)
