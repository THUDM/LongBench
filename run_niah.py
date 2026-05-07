"""
This script is adapted from
https://github.com/FranxYao/Long-Context-Data-Engineering
"""

import tiktoken
import os
import pdb
import glob
import jieba

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from anthropic import Anthropic
import numpy as np
import argparse
from rouge_score import rouge_scorer

import sys
import os

from openai import OpenAI
from datetime import datetime, timezone
import time
import torch

from models.compression.monkeypatch import replace_qwen3_5

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
LOCAL_MODEL_PROVIDERS = ["LLaMA3", "Mistral", "Qwen", "Qwen3.5", "HF"]


def build_compression_config(compression_mode, compression_budget):
    return {
        "method": compression_mode,
        "method_config": {
            "budget": compression_budget,
            "window_size": 8,
            "mix_lambda": 0.07,
            "retain_ratio": 0.2,
            "retain_direction": "last",
            "first_tokens": 4,
        },
        "compression": None,
        "update_kv": True,
    }


def apply_qwen3_5_compression_setup(model, tokenizer, compression_mode):
    model.config.update(
        {
            "divide_method": "step_length",
            "divide_length": 128,
            "compression_content": "think",
            "method": compression_mode,
        }
    )
    model.newline_token_ids = [
        tokenizer.encode(text, add_special_tokens=False)[-1]
        for text in ["\n", ".\n", ")\n", "\n\n", ".\n\n", ")\n\n"]
    ]
    model.after_think_token_ids = [tokenizer.encode("</think>", add_special_tokens=False)[-1]]


def load_model_and_tokenizer(
    model_path,
    attn_implementation="flash_attention_2",
    use_fast_tokenizer=True,
    compression=False,
    compression_mode=None,
    compression_budget=4096,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=use_fast_tokenizer,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "attn_implementation": attn_implementation,
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model_path_lower = model_path.lower()
    if compression:
        if not compression_mode:
            raise ValueError("Please provide --compression_mode when --compression is enabled.")
        if "qwen3.5" not in model_path_lower:
            raise ValueError(
                f"Compression currently supports only qwen3.5 models, got: {model_path}"
            )

        replace_qwen3_5(build_compression_config(compression_mode, compression_budget))
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        apply_qwen3_5_compression_setup(model, tokenizer, compression_mode)
    elif "qwen3.5" in model_path_lower:
        from models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
        from models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

        config = Qwen3_5Config.from_pretrained(model_path)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            **model_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    model.eval()
    return model, tokenizer


def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_inputs(prompt, tokenizer, device, enable_thinking=False):
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            inputs = input_ids if isinstance(input_ids, dict) else {"input_ids": input_ids}
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    inputs = dict(inputs)
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return {k: v.to(device) for k, v in inputs.items()}


def validate_args(args):
    if not args.model_name:
        raise ValueError("--model_name is required.")
    if args.compression and not args.compression_mode:
        raise ValueError("--compression requires --compression_mode.")
    if args.compression and args.compression_budget < 1:
        raise ValueError("--compression_budget must be at least 1 when compression is enabled.")

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="data/PaulGrahamEssays", # PaulGrahamEssays
                 retrieval_question="The best thing to do in San Francisco is: ",
                 results_version = 1,
                 context_lengths_min = None,
                 context_lengths_max = None,
                 context_lengths_num_intervals = 40,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 model_provider = "OpenAI",
                 openai_api_key=None,
                 anthropic_api_key = None,
                 model_name='',
                 model_name_suffix=None,
                 model_version=None,
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True,
                 step=100,
                 attn_implementation='flash_attention_2',
                 compression=False,
                 compression_mode=None,
                 compression_budget=4096,
                 enable_thinking=False):
        """
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 0.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")

        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        self.step = step
        self.attn_implementation = attn_implementation
        self.compression = compression
        self.compression_mode = compression_mode
        self.compression_budget = compression_budget
        self.enable_thinking = enable_thinking


        self.model_version = model_version or os.path.basename(model_name.rstrip("/"))
        if(model_name_suffix is not None): self.model_version += "_" + model_name_suffix
        if compression:
            self.model_version += f"_{compression_mode}_{compression_budget}"

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                # self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
                self.context_lengths = np.arange(context_lengths_min, context_lengths_max+1, step=self.step)
        else:
            self.context_lengths = context_lengths


        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")

        self.model_name = model_name

        if self.model_provider not in LOCAL_MODEL_PROVIDERS:
            raise ValueError("model_provider must be one of 'LLaMA3', 'Mistral', 'Qwen', 'Qwen3.5', or 'HF'")

        print("loading from %s" % model_name)
        self.model_to_test, self.enc = load_model_and_tokenizer(
            model_name,
            attn_implementation=self.attn_implementation,
            compression=self.compression,
            compression_mode=self.compression_mode,
            compression_budget=self.compression_budget,
        )


    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self, args):

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len: continue
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(context_length, depth_percent)

    def generate_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure
        if(self.model_provider not in ["OpenAI", "Anthropic"]):
            test_format=f"This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
            return test_format
        else:
            return [
                {
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                },
                {
                    "role": "user",
                    "content": context
                    },
                {
                    "role": "user",
                    "content": f"{self.retrieval_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer, and I'm 100% sure. So try your best to find it."
                },
                {
                    "role": "assistant",
                    "content":"",
                },

            ]

    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                print("result exists, skipping")
                return
            else:
                print("result does not exist, testing")

        # Go generate the required length context and place your needle statement in
        context = self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context)
        test_start_time = time.time()

        if(self.model_provider in LOCAL_MODEL_PROVIDERS):

            inputs = build_inputs(
                prompt,
                self.enc,
                get_input_device(self.model_to_test),
                enable_thinking=self.enable_thinking,
            )
            input_ids = inputs["input_ids"]
            eos_token_ids = [
                token_id
                for token_id in [
                    self.enc.eos_token_id,
                    self.enc.encode("\n", add_special_tokens=False)[-1],
                ]
                if token_id is not None
            ]
            output_ids = self.model_to_test.generate(
                **inputs,
                output_attentions=False,
                max_new_tokens=30,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.enc.pad_token_id,
                eos_token_id=eos_token_ids,
            )
            response = self.enc.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


        print(response)
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        if len(response) != 0:
            # score = scorer.score(self.needle, response)['rouge1'].fmeasure*10
            score = scorer.score("eat a sandwich and sit in Dolores Park on a sunny day", response)['rouge1'].fmeasure*10
        else:
            score = 0.0

        results = {
            'model' : self.model_name,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'),
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            results['file_name'] : context_file_location

            # Save the context to file for retesting
            if not os.path.exists('results_needle/contexts'):
                os.makedirs('results_needle/contexts')

            if not os.path.exists(f'results_needle/contexts/{self.model_version}'):
                os.makedirs(f'results_needle/contexts/{self.model_version}')

            with open(f'results_needle/contexts/{self.model_version}/{context_file_location}_context.txt', 'w') as f:
                f.write(context)

        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists('results_needle/results'):
                os.makedirs('results_needle/results')

            if not os.path.exists(f'results_needle/results/{self.model_version}'):
                os.makedirs(f'results_needle/results/{self.model_version}')

            # Save the result to file for retesting
            p = f'results_needle/results/{self.model_version}/{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f, ensure_ascii=False)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'output_dir/results_needle/results/' + self.model_version
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context

    def encode_text_to_tokens(self, text):
        if self.model_provider in LOCAL_MODEL_PROVIDERS:
            return self.enc.encode(text, add_special_tokens=False)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            return self.enc.encode(text)
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer
        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]
            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            if(self.model_provider in ["LLaMA", "LongLLaMA"]): period_tokens = [29889, 869]
            elif(self.model_provider == "LLaMA3"): period_tokens = [13]
            elif(self.model_provider == "Mistral"): period_tokens = [29473]  # [842, 28723]
            elif(self.model_provider == "GLM"): period_tokens = [918, 30930]
            else: period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        if self.model_provider in LOCAL_MODEL_PROVIDERS:
            return len(self.enc.encode(context, add_special_tokens=False))
        else:
            return len(self.enc.encode(context))
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider in LOCAL_MODEL_PROVIDERS:
            return self.enc.encode(context, add_special_tokens=False)
        else:
            return self.enc.encode(context)
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider in LOCAL_MODEL_PROVIDERS:
            return self.enc.decode(tokens[:context_length], skip_special_tokens=True)
        else:
            return self.enc.decode(tokens[:context_length])
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context

    def get_results(self):
        return self.testing_results

    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        #asyncio.run(self.run_test())
        self.run_test(args)


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument('--model_version', type=str, default=None, help='provider of model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="HF", help='which model to use')
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('--step', type=int, default=1000)
    parser.add_argument("--compression", action="store_true")
    parser.add_argument("--compression_mode", type=str, default=None)
    parser.add_argument("--compression_budget", type=int, default=4096)
    parser.add_argument("--enable_thinking", action="store_true", help="Pass enable_thinking=True to chat templates that support it.")
    args = parser.parse_args()


    validate_args(args)

    ht = LLMNeedleHaystackTester(model_name=args.model_name,
                                 model_name_suffix=args.model_name_suffix,
                                 model_provider=args.model_provider,
                                 model_version=args.model_version,
                                 context_lengths_min=args.s_len,
                                 save_contexts=True,
                                 save_results=True,
                                 openai_api_key=args.api_key,
                                 context_lengths_max=args.e_len,
                                 step=args.step,
                                 attn_implementation=args.attn_implementation,
                                 compression=args.compression,
                                 compression_mode=args.compression_mode,
                                 compression_budget=args.compression_budget,
                                 enable_thinking=args.enable_thinking
                                 )

    ht.start_test(args)
