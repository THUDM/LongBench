import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Callable
from transformers.utils import logging
from transformers.processing_utils import Unpack

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5RMSNorm,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_rotary_pos_emb
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    eager_attention_forward
)
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs


from .methods import (
    SnapKV,
    StreamingLLM,
    H2O,
    GateKV,
)

import math
import torch.nn.functional as F

KV_COMPRESSION_MAP = {
    "snapkv": SnapKV,
    "streamingllm": StreamingLLM,
    "h2o": H2O,
    "gatekv": GateKV,
}

logger = logging.get_logger(__name__)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _compress_gate_cache(gate_cache, kept_indices, num_key_value_groups):
    gate_cache = gate_cache.transpose(1, 2)
    if num_key_value_groups != 1:
        kept_indices = kept_indices.repeat_interleave(num_key_value_groups, dim=1)
    kept_indices = kept_indices.to(device=gate_cache.device, dtype=torch.long)
    kept_indices = kept_indices.unsqueeze(-1).expand(
        -1,
        -1,
        -1,
        gate_cache.shape[-1],
    )
    return gate_cache.gather(dim=2, index=kept_indices).transpose(1, 2)


def Qwen3_5Attention_init(
    self, config: Qwen3_5Config, layer_idx: int, compression_config: dict
):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    self.num_key_value_groups = (
        config.num_attention_heads // config.num_key_value_heads
    )
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True
    self.q_proj = nn.Linear(
        config.hidden_size,
        config.num_attention_heads * self.head_dim * 2,
        bias=config.attention_bias,
    )
    self.k_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.v_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.o_proj = nn.Linear(
        config.num_attention_heads * self.head_dim,
        config.hidden_size,
        bias=config.attention_bias,
    )
    self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    # =============== New logic start ===============
    self.config.update(compression_config)
    self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
        layer_idx=self.layer_idx,
        model_config=self.config,
        model_type="qwen3",
        **compression_config["method_config"],
    )
    # =============== New logic end =================

def Qwen3_5Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate_states = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
        2,
        dim=-1,
    )
    gate = gate_states.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_values is not None:
        layer_cache = past_key_values.layers[self.layer_idx]
        # =============== Enable Query Cache ============
        if not hasattr(layer_cache, "query_cache"):
            layer_cache.query_cache = None

        if layer_cache.query_cache is None:
            bsz, n_heads, _, head_dim = query_states.shape
            layer_cache.query_cache = torch.empty(
                bsz, n_heads, 0, head_dim
            )
            layer_cache.query_cache = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
        else:
            layer_cache.query_cache = torch.cat(
                (layer_cache.query_cache, query_states),
                dim=2,
            )

            window_size = self.config.method_config["window_size"]
            if layer_cache.query_cache.shape[-2] > window_size:
                layer_cache.query_cache = layer_cache.query_cache[
                    :, :, -window_size:, :
                ]
        # =============== Enable Query Cache end =========

        requires_gate_states = getattr(self.kv_cluster, "requires_gate_states", False)
        if requires_gate_states:
            if not hasattr(layer_cache, "gate_cache") or layer_cache.gate_cache is None:
                layer_cache.gate_cache = gate_states
            else:
                layer_cache.gate_cache = torch.cat(
                    (layer_cache.gate_cache, gate_states),
                    dim=1,
                )

        # =============== decoding-time compression start ===============
        cached_queries = layer_cache.query_cache
        if self.config.compression is None or query_states.shape[-2] > 1:
            update_kwargs = {}
            if requires_gate_states:
                update_kwargs = {
                    "gate_states": layer_cache.gate_cache,
                    "is_prefill": query_states.shape[-2] > 1,
                }
            elif getattr(self.kv_cluster, "record_gate_overlap", False):
                update_kwargs = {
                    "gate_states": gate_states,
                    "is_prefill": query_states.shape[-2] > 1,
                }
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,
                value_states,
                **update_kwargs,
            )

            if self.config.update_kv is True:
                past_key_values.update(
                    key_states_compress,
                    value_states_compress,
                    self.layer_idx,
                )
                if requires_gate_states:
                    kept_indices = getattr(self.kv_cluster, "last_kept_indices", None)
                    if kept_indices is not None:
                        layer_cache.gate_cache = _compress_gate_cache(
                            layer_cache.gate_cache,
                            kept_indices,
                            self.num_key_value_groups,
                        )
            else:
                past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                )

        elif self.config.compression is True:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
            )

            update_kwargs = {}
            if requires_gate_states:
                update_kwargs = {
                    "gate_states": layer_cache.gate_cache,
                    "is_prefill": query_states.shape[-2] > 1,
                }
            elif getattr(self.kv_cluster, "record_gate_overlap", False):
                update_kwargs = {
                    "gate_states": gate_states,
                    "is_prefill": query_states.shape[-2] > 1,
                }
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,
                value_states,
                **update_kwargs,
            )

            if self.config.update_kv is True:
                layer_cache.keys = key_states_compress
                layer_cache.values = value_states_compress
                if requires_gate_states:
                    kept_indices = getattr(self.kv_cluster, "last_kept_indices", None)
                    if kept_indices is not None:
                        layer_cache.gate_cache = _compress_gate_cache(
                            layer_cache.gate_cache,
                            kept_indices,
                            self.num_key_value_groups,
                        )
        else:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
            )
        # =============== decoding-time compression end ===============

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation,
        eager_attention_forward,
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def Qwen3_5ForCausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    
    # sample-level statistics
    if past_key_values.get_seq_length() == 0:
        if self.config.compression_content == "think":
            self.after_think = False

    if not hasattr(self, "length"):
        self.length = input_ids.shape[1]
    else:
        self.length += input_ids.shape[1]

    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    # =============== Step-level Compression logic start ===============
    # assume non-batch input, shape: [1, logits_to_keep, vocab_size]
    predicted_token_ids = logits[:, -1, :].argmax(dim=-1)

    if self.config.compression_content == "think" and self.after_think == False:
        self.after_think = (
            predicted_token_ids[0].cpu().item() in self.after_think_token_ids
        )

    if self.config.divide_method == "newline":
        is_newline = predicted_token_ids[0].cpu().item() in self.newline_token_ids
    elif self.config.divide_method == "step_length":
        is_newline = self.length % self.config.divide_length == 0
    else:
        raise ValueError(f"Invalid divide_method: {self.config.divide_method}")

    if self.config.compression_content == "think" and self.after_think == True:
        is_newline = False

    # Set compression flag for all layers at once
    for layer in self.model.layers:
        if layer.layer_type == "full_attention":
            layer.self_attn.config.compression = is_newline
    # =============== Step-level Compression logic end =================

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.vocab_size,
            **kwargs,
        )

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def Qwen3_5ForConditionalGeneration_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs,
) -> Union[Tuple, Qwen3_5CausalLMOutputWithPast]:
    
    # sample-level statistics
    if past_key_values.get_seq_length() == 0:
        if self.config.compression_content == "think":
            self.after_think = False

    if not hasattr(self, "length"):
        self.length = input_ids.shape[1]
    else:
        self.length += input_ids.shape[1]

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        mm_token_type_ids=mm_token_type_ids,
        **kwargs,
    )

    hidden_states = outputs[0]
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    # =============== Step-level Compression logic start ===============
    # assume non-batch input, shape: [1, logits_to_keep, vocab_size]
    predicted_token_ids = logits[:, -1, :].argmax(dim=-1)

    if self.config.compression_content == "think" and self.after_think == False:
        self.after_think = (
            predicted_token_ids[0].cpu().item() in self.after_think_token_ids
        )

    if self.config.divide_method == "newline":
        is_newline = predicted_token_ids[0].cpu().item() in self.newline_token_ids
    elif self.config.divide_method == "step_length":
        is_newline = self.length % self.config.divide_length == 0
    else:
        raise ValueError(f"Invalid divide_method: {self.config.divide_method}")

    if self.config.compression_content == "think" and self.after_think == True:
        is_newline = False

    # Set compression flag for all layers at once
    for layer in self.model.layers:
        if layer.layer_type == "full_attention":
            layer.self_attn.config.compression = is_newline
    # =============== Step-level Compression logic end =================

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.text_config.vocab_size,
        )

    return Qwen3_5CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )
