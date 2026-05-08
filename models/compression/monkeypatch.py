from transformers.models.qwen3_5 import modeling_qwen3_5
from .modeling import (
    Qwen3_5Attention_init,
    Qwen3_5Attention_forward,
    Qwen3_5GatedDeltaNet_forward,
    Qwen3_5ForCausalLM_forward,
    Qwen3_5ForConditionalGeneration_forward,
)

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .flash_attn.flash_attention import flash_attention_forward

def replace_qwen3_5(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen3_5Attention_init(self, config, layer_idx, compression_config)

    original_gated_delta_net_init = getattr(
        modeling_qwen3_5.Qwen3_5GatedDeltaNet,
        "_compression_original_init",
        modeling_qwen3_5.Qwen3_5GatedDeltaNet.__init__,
    )
    modeling_qwen3_5.Qwen3_5GatedDeltaNet._compression_original_init = (
        original_gated_delta_net_init
    )

    def gated_delta_net_init_wrapper(self, config, layer_idx):
        original_gated_delta_net_init(self, config, layer_idx)
        self.compression_method_config = compression_config.get("method_config", {})

    modeling_qwen3_5.Qwen3_5Attention.__init__ = init_wrapper
    modeling_qwen3_5.Qwen3_5Attention.forward = Qwen3_5Attention_forward
    modeling_qwen3_5.Qwen3_5GatedDeltaNet.__init__ = gated_delta_net_init_wrapper
    modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward = Qwen3_5GatedDeltaNet_forward
    modeling_qwen3_5.Qwen3_5ForCausalLM.forward = Qwen3_5ForCausalLM_forward
    modeling_qwen3_5.Qwen3_5ForConditionalGeneration.forward = (
        Qwen3_5ForConditionalGeneration_forward
    )

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
