import requests
import json
import jsonlines
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import os
from threading import Lock
import time
import re
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, default=500)
parser.add_argument('--model', type=str, default="Llama2") # glm2, gpt-16k, Llama2
args = parser.parse_args()
print(args)
GPT_key = "" #openai key
GPT_MODEL = 'gpt-3.5-turbo-16k'
GLM_MODEL = "THUDM/chatglm2-6b-32k"
LLAMA_MODEL = "meta-llama/Llama-2-7b-chat-hf"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

jsonl_files = ['qmsum.jsonl', 'gov_report.jsonl',  'vcsum.jsonl','multinews.jsonl']
raw_data_folder = '../LongBench/data' #raw data folder
new_data_folder = args.model+'_'+str(args.max_len)+'/data' #compressed data folder
compressed_context_path ='../LongBench/compressed_data_'+str(args.max_len)+'/data' #compressed context folder 
if not os.path.exists(new_data_folder):
    os.makedirs(new_data_folder)
if not os.path.exists(compressed_context_path):
    os.makedirs(compressed_context_path)

def build_chat(tokenizer, prompt, model_name):
    if "glm2" in model_name:
        prompt = tokenizer.build_prompt(prompt)     
    elif "Llama2" in model_name:
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

if args.model=="glm2":
    model_path = GLM_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,max_length=1024)
    # model = load_model_on_gpus(model_path, num_gpus=4)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    def generate_response(prompt):
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        max_length = 31500
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        prompt = build_chat(tokenizer, prompt, 'glm2')
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        output = model.generate(
            **input,
            max_new_tokens=200,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        return pred
        
elif args.model=="gpt-16k":
    #using GPT API and tokenizer
    def query(messages, force_commit=False):
        tries = 0
        while tries < 5:
            tries += 1
            try:
                headers = {
                    'Authorization': GPT_key
                }
                resp = requests.post("https://api.openai.com/v1/chat/completions", json = {
                    "model": GPT_MODEL,
                    "messages": messages,
                    "temperature": 1.0
                }, headers=headers, timeout=120)
                if resp.status_code != 200:
                    raise Exception(resp.text)
                resp = resp.json()
                break
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                if "maximum context length" in str(e):
                    raise e
                print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
        else:
            print("Max tries. Failed.")
            return
        return resp["choices"][0]["message"]["content"]
    
    def generate_response(prompt):
        msg = [{"role": "user", "content": prompt}]
        result = query(msg)
        return result
    
elif args.model=="Llama2":
    replace_llama_attn_with_flash_attn()
    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL)
    model =LlamaForCausalLM.from_pretrained(LLAMA_MODEL, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    def generate_response(prompt):
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        max_length = 3500
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            prompt = build_chat(tokenizer, prompt, 'Llama2')
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        output = model.generate(
            **input,
            max_new_tokens=200,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)  
        return pred
    
def get_word_list(s1):

    regEx = re.compile('[\W]')   
    res = re.compile(r"([\u4e00-\u9fa5])")    #  [\u4e00-\u9fa5] for zh

    p1 = regEx.split(s1.lower())
    str1_list = []
    for str in p1:
        if res.split(str) == None:
            str1_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str1_list.append(ch)

    list_word1 = [w for w in str1_list if len(w.strip()) > 0]  
    return  list_word1

def get_word_len(s1):
    return len(get_word_list(s1))    

def data_spilt(data_test,max_len=args.max_len):

    data_len=len(data_test)
    #split data_test to n parts averagely according to data_len
    data_words_len = get_word_len(data_test)
    split_len = int(data_words_len/max_len)

    text_len = int(data_len/split_len)
    #start position of each part
    text = data_test
    text_list=[]
    text_num = 0
    while text_num <= split_len:
        text_num += 1
        try:
            for i in range(text_len,text_len+1500):
        
                    if text[i] in ['\n', '.', '。', '?', '？', '!', '！']:
                        # cut off the text until the end of the line, using the length of the text to decide when to stop   
                        text_list.append(text[:i+1])
                        text = text[i+1:]
                        break
        except:
            if text not in text_list:
                text_list.append(text)
    if text!='' and text not in text_list:
        text_list.append(text)

    return text_list

def compress(data_test, max_len=args.max_len, language='en',_id=None,dataset_type=None):
    try:
        text_list = data_spilt(data_test)
        responses = []
        compressed_data_list = []  # List to store compressed data
        # compressed_context_path = 
        compressed_id = 0  # Counter for compressed_id
        prompt_en = 'Summerize the context above. The max length of the summary is 200 words.'
        prompt_zh = '请对上面的文本进行总结。最大长度为200个字。'
        for text_num, text in enumerate(text_list):
            # Your existing code to generate prompt based on language
            if language == 'zh':
                prompt = text + '\n' + prompt_zh
            elif language == 'en':
                prompt = text + '\n' + prompt_en
            else:
                prompt = text + '\n' + prompt_en
                print('language error')
            response = generate_response(prompt)
            responses.append(response)
            compressed_context = response  # Change this based on your response generation logic
            # Construct the compressed data entry
            compressed_entry = {
                "input": "",  # Fill in the input/command here
                "raw_context": text,  # Original text
                "compressed_context": compressed_context,  # Compressed text
                "compressed_id": compressed_id,
                "answers": [],  # Fill in the answers
                "length": get_word_len(text),  # Length of the original text
                "dataset": str(dataset_type),  # Fill in the dataset name
                "language": language,
                "all_classes": None,  # Fill in the categories
                "_id": _id  # Fill in the ID
            }

            compressed_data_list.append(compressed_entry)
            compressed_id += 1
        
        # Save the compressed data to a file
        path = os.path.join(compressed_context_path, str(args.model)+'_'+str(max_len)+'_'+str(dataset_type)+'.jsonl')
        with jsonlines.open(path, 'a') as writer:
            writer.write_all(compressed_data_list)
        #save repsonses to file
        if language == 'en':
            for i in range(len(responses)):
                responses[i] = 'Paragraph Summary'+str(i+1)+" : "+responses[i]+'\n'
        elif language == 'zh':
            for i in range(len(responses)):
                responses[i] = '段落摘要'+str(i+1)+" : "+responses[i]+'\n'
            
        new_text_words_len =get_word_len(new_text)
        print('new_text_words_len :',new_text_words_len)
    except Exception as e:
        print(str(e), sep=" | ")
        new_text = data_test
    return new_text      

def handle_item(item, max_len):
    try:
        context = item['context']  
        language = item['language']
        _id = item['_id']
        dataset_type = item['dataset']
        compressd_context = compress(context, max_len,language,_id,dataset_type)
        item['context'] = compressd_context
        if 'all_classes' not in item.keys():
            item['all_classes'] = 'None'
        #count the length of compressd context basen on language
        item['length'] = get_word_len(compressd_context)

    except Exception as e:
        print('Compress Fail：',str(e), sep=" | ")
    return item

def save_data(data, file_name):
    with jsonlines.open(file_name, 'a') as writer:
        writer.write_all(data)
    return data

def load_data(file_name):
    with jsonlines.open(file_name) as reader:
        data = list(reader)
    return data

def parallel_process_data(data, start_index, handle_item, workers=20, callback=None, checkpoint_interval=20):
    save_lock = Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        id_set = set()
        for i in range(start_index):
            id_set.add(data[i]['_id'])
        
        for i in range(start_index, len(data)):
            item = data[i]
            future = executor.submit(handle_item, item, args.max_len)
            futures.append(future)
            if i > 0 and i % checkpoint_interval == 0:

                processed_data = []
                
                with tqdm(total=len(futures)) as pbar:
                    for future in concurrent.futures.as_completed(futures):

                        result = future.result()
                        if result['_id'] not in id_set:
                            id_set.add(result['_id'])
                            processed_data.append(result)
                        if callback:
                            callback(result)
                        pbar.update(1)

                with save_lock:
                    save_data(processed_data, new_file_path)
                    save_data(processed_data, checkpoint_file)

        processed_data = []
        print('final save data')
        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result['_id'] not in id_set:
                    id_set.add(result['_id'])
                    processed_data.append(result)
                if callback:
                    callback(result)
                pbar.update(1)

        with save_lock:
            try:
                save_data(processed_data, new_file_path)
                save_data(processed_data, checkpoint_file)
            except Exception as e:
                time.sleep(1)
                print('Save data：',str(e), sep=" | ")
                save_data(processed_data, new_file_path)
                save_data(processed_data, checkpoint_file)

for jsonl_file in jsonl_files:
    raw_file_path = os.path.join(raw_data_folder, jsonl_file)
    print('load data from', raw_file_path)
    data = load_data(raw_file_path)
    print('start compress')
    checkpoint_folder = os.path.join(new_data_folder, 'checkpoint')
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    checkpoint_file = os.path.join(checkpoint_folder, jsonl_file + '_checkpoint.jsonl')
    start_index = 0
    if os.path.exists(checkpoint_file):
        processed_data = load_data(checkpoint_file)
        start_index = len(processed_data)
        print('load checkpoint from', checkpoint_file, 'start_index:', start_index)
    new_file_path = os.path.join(new_data_folder, jsonl_file)
    parallel_process_data(data, start_index, handle_item, workers=19, checkpoint_interval=19)
    print('end compress')
    print('save data to', new_file_path)