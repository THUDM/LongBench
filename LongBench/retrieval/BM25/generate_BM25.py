from rank_bm25 import BM25Okapi

import os
import json
from concurrent.futures import ThreadPoolExecutor, wait
from tqdm import tqdm
import argparse
import sys
sys.path.append('..')
from splitter import split_long_sentence, get_word_len, regex
# DEBUG
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

def retriveDoc(query: str, document: str, chunk_size, file_name:str,
               js, output_list, idx, pbar=None, maxLen=1500):
    # 1. Splits the context into pieces
    texts = split_long_sentence(document, regex, chunk_size=chunk_size, filename=file_name)
    # 2. Creates retriver, adds texts
    retriever = BM25Okapi(texts)
    # 3. Retrive and merge
    retrieved_texts = retriever.get_top_n(query=query, documents=texts,
        n=len(texts))
    retrieved_texts = [retrieved_texts] if type(retrieved_texts) == str else retrieved_texts
    context = ''
    for text in retrieved_texts:
        if get_word_len(context) < maxLen:
            context += text
    js['retrieved'] = retrieved_texts if type(retrieved_texts) == list else [retrieved_texts]
    js['context'] = context
    js['length'] = get_word_len(context)
    output_list[index] = js
    if pbar:
        pbar.update()
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--file_name", default='dureader.jsonl')
    parser.add_argument("--file_name", default='2wikimqa.jsonl')
    parser.add_argument("--source_dir", default='../../LongBench/data')
    parser.add_argument("--dest_dir", default='./test')
    parser.add_argument("--chunk_size", type=int, default=200)
    args = parser.parse_args()
    file_name = args.file_name

    print(f"------  {file_name}  ------")
    with open(os.path.join(args.source_dir, file_name), 'r', encoding='utf-8') as file:
        file_contents = file.readlines()
        output_data = [{}] * len(file_contents)
        if (os.path.exists(os.path.join(args.dest_dir, file_name))):
            with open(os.path.join(args.dest_dir, file_name), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                lines = [line for line in lines]
                output_data = [json.loads(line) for line in lines]
        loop = tqdm(enumerate(file_contents), total=len(file_contents), desc=f'{file_name}')
        exe_list = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            # for index, line in loop:
            for index, line in enumerate(file_contents):
                if (output_data[index] != {} or
                    "context" in output_data[index].keys() and len(output_data[index]['context']) != 0):
                    loop.update()
                    continue
                line_js = json.loads(line)
                retriveDoc(query=line_js['input'], document=line_js['context'],
                                        chunk_size=args.chunk_size, file_name=file_name,
                                        js=line_js, output_list=output_data, idx=index, pbar=loop)
                # exe_list.append(executor.submit(retriveDoc, query=line_js['input'], document=line_js['context'],
                #                         chunk_size=args.chunk_size, file_name=file_name,
                #                         js=line_js, output_list=output_data, idx=index, pbar=loop))
                # loop.set_description(f'{file_name}')
                wait(exe_list)
    # saving
    os.makedirs(args.dest_dir, exist_ok=True)
    with open(os.path.join(args.dest_dir, file_name), 'w', encoding='utf-8') as output_file:
        for item in output_data:
            output_file.write(json.dumps(item, ensure_ascii=False) + '\n')
