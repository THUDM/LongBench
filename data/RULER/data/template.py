# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Templates = {
    'base': "{task_template}",

    'meta-chat': "[INST] {task_template} [/INST]",

    'vicuna-chat': "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {task_template} ASSISTANT:",

    'lwm-chat': "You are a helpful assistant. USER: {task_template} ASSISTANT: ",

    'command-r-chat': "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{task_template}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",

    'chatglm-chat': "[gMASK]sop<|user|> \n {task_template}<|assistant|> \n ",
    
    'RWKV': "User: hi\n\nAssistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it\n\nUser: {task_template}\n\nAssistant:",

    'Phi3': "<|user|>\n{task_template}<|end|>\n<|assistant|>\n",

    'meta-llama3': "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{task_template}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    
    'jamba': "<|startoftext|><|bom|><|system|> <|eom|><|bom|><|user|> {task_template}<|eom|><|bom|><|assistant|>",
    
    'nemotron5-instruct': "<SPECIAL_10>System\n\n<SPECIAL_11>User\n{task_template}\n<SPECIAL_11>Assistant\n",
}

APPLY_CHAT_TEMPLATE_TYPES = {"auto", "apply_chat_template", "chat_template"}


def _get_hf_tokenizer(tokenizer):
    return getattr(tokenizer, "tokenizer", tokenizer)


def render_model_template(tokenizer, task_template, model_template_type):
    if model_template_type == "base":
        return Templates["base"].format(task_template=task_template)

    if model_template_type not in Templates and model_template_type not in APPLY_CHAT_TEMPLATE_TYPES:
        raise ValueError(
            f"{model_template_type} is not found in {list(Templates.keys()) + sorted(APPLY_CHAT_TEMPLATE_TYPES)}"
        )

    hf_tokenizer = _get_hf_tokenizer(tokenizer)
    if not hasattr(hf_tokenizer, "apply_chat_template"):
        raise ValueError(
            f"Tokenizer for model_template_type={model_template_type} does not support apply_chat_template."
        )

    messages = [{"role": "user", "content": task_template}]
    try:
        return hf_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return hf_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
