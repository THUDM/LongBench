from transformers.models.llama import modeling_llama
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers.models.qwen3_5 import modeling_qwen3_5
from .modeling import (
    Llama_Attention_init,
    Llama_Attention_forward,
    Llama_CausalLM_forward,
    Qwen3_Attention_init,
    Qwen3_Attention_forward,
    Qwen3_CausalLM_forward,
    Qwen3Moe_Attention_init,
    Qwen3Moe_Attention_forward,
    Qwen3Moe_CausalLM_forward,
    Qwen3_5Attention_init,
    Qwen3_5Attention_forward,
    Qwen3_5ForConditionalGeneration_forward,
)

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .flash_attn.flash_attention import flash_attention_forward

def replace_llama(compression_config):
    def init_wrapper(self, config, layer_idx):
        Llama_Attention_init(self, config, layer_idx, compression_config)

    modeling_llama.LlamaAttention.__init__ = init_wrapper
    modeling_llama.LlamaAttention.forward = Llama_Attention_forward
    modeling_llama.LlamaForCausalLM.forward = (
        Llama_CausalLM_forward
    )

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward

def replace_qwen3(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen3_Attention_init(self, config, layer_idx, compression_config)

    modeling_qwen3.Qwen3Attention.__init__ = init_wrapper
    modeling_qwen3.Qwen3Attention.forward = Qwen3_Attention_forward
    modeling_qwen3.Qwen3ForCausalLM.forward = (
        Qwen3_CausalLM_forward
    )

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward

def replace_qwen3moe(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen3Moe_Attention_init(self, config, layer_idx, compression_config)

    modeling_qwen3_moe.Qwen3MoeAttention.__init__ = init_wrapper
    modeling_qwen3_moe.Qwen3MoeAttention.forward = Qwen3Moe_Attention_forward
    modeling_qwen3_moe.Qwen3MoeForCausalLM.forward = (
        Qwen3Moe_CausalLM_forward
    )

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward

def replace_qwen3_5(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen3_5Attention_init(self, config, layer_idx, compression_config)

    modeling_qwen3_5.Qwen3_5Attention.__init__ = init_wrapper
    modeling_qwen3_5.Qwen3_5Attention.forward = Qwen3_5Attention_forward
    modeling_qwen3_5.Qwen3_5ForConditionalGeneration.forward = (
        Qwen3_5ForConditionalGeneration_forward
    )

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
