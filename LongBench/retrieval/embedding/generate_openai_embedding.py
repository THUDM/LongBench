import openai
from openai.embeddings_utils import cosine_similarity
openai.api_key="KEY"
openai.proxy=""

import os
import json
from concurrent.futures import ThreadPoolExecutor, wait
from tqdm import tqdm
import argparse
import sys
sys.path.append("..")
from splitter import split_long_sentence, get_word_len, regex
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

def retriveDoc(query: str, document: str, chunk_size, file_name:str,
               js, output_list, idx, pbar=None, maxLen=1500):
    # 1. Splits the context into pieces
    texts = split_long_sentence(document, regex, chunk_size=chunk_size, filename=file_name)
    # 2. Creates retriver, adds texts
    # 3. Retrives and merges
    # https://platform.openai.com/docs/api-reference/embeddings/object?lang=python
    texts_embeddings = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    query_embeddings = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=query
    )
    similarity = []
    for emb in texts_embeddings['data']:
        similarity.append(cosine_similarity(emb['embedding'], query_embeddings['data'][0]['embedding']))
    sorted_pairs=sorted(zip(similarity, texts), reverse=True)
    retrieved_texts = [pair[1] for pair in sorted_pairs]
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
    parser.add_argument("--file_name", default='musique.jsonl')
    parser.add_argument("--source_dir", default='../source/docqa_only')
    parser.add_argument("--dest_dir", default='./test')
    parser.add_argument("--chunk_size", type=int, default=200)
    args = parser.parse_args()
    file_name = args.file_name

    print(f"------  {file_name}  ------")
    with open(os.path.join(args.source_dir, file_name), 'r', encoding='utf-8') as file:
        file_contents = file.readlines()
        # DEBUG
        # file_contents = file_contents[:10]
        # with tqdm(total=len(file_contents)) as pbar, ThreadPoolExecutor(max_workers=1) as executor:
        output_data = [{}] * len(file_contents)
        if (os.path.exists(os.path.join(args.dest_dir, file_name))):
            with open(os.path.join(args.dest_dir, file_name), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                lines = [line for line in lines]
                output_data = [json.loads(line) for line in lines]
        def saving():
            os.makedirs(args.dest_dir, exist_ok=True)
            with open(os.path.join(args.dest_dir, file_name), 'w', encoding='utf-8') as output_file:
                for item in output_data:
                    output_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        loop = tqdm(enumerate(file_contents), total=len(file_contents), desc=f'{file_name}')
        exe_list = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            # for index, line in loop:
            for index, line in enumerate(file_contents):
                if (output_data[index] != {} or
                    "context" in output_data[index].keys() and len(output_data[index]['context']) != 0):
                    loop.update()
                    continue
                line_js = json.loads(line)
                
                try:
                    # retriveDoc(query=line_js['input'], document=line_js['context'],
                    #                     chunk_size=args.chunk_size, file_name=file_name,
                    #                     js=line_js, output_list=output_data, idx=index, pbar=loop)
                    exe_list.append(executor.submit(retriveDoc, query=line_js['input'], document=line_js['context'],
                                            chunk_size=args.chunk_size, file_name=file_name,
                                            js=line_js, output_list=output_data, idx=index, pbar=loop))
                except Exception as e:
                    saving()
                    print(e)
                wait(exe_list)
        saving()
    
