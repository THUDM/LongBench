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
    apply_mask_to_padding_states,
    apply_rotary_pos_emb,
    l2norm,
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


def _compress_linear_state_cache(linear_state_cache, kept_indices):
    kept_indices = kept_indices.to(device=linear_state_cache.device, dtype=torch.long)
    return linear_state_cache.gather(dim=2, index=kept_indices)


def _align_linear_state_heads(linear_state_scores, num_kv_heads):
    batch_size, num_linear_heads, seq_len = linear_state_scores.shape
    if num_linear_heads == num_kv_heads:
        return linear_state_scores
    if num_linear_heads % num_kv_heads == 0:
        group_size = num_linear_heads // num_kv_heads
        return linear_state_scores.view(
            batch_size,
            num_kv_heads,
            group_size,
            seq_len,
        ).max(dim=2).values

    return linear_state_scores.mean(dim=1, keepdim=True).expand(
        batch_size,
        num_kv_heads,
        seq_len,
    )


def _parse_linear_state_layer_range(layer_range):
    if layer_range is None or layer_range == "all":
        return 1, None
    if isinstance(layer_range, int):
        if layer_range < 1:
            raise ValueError("linear_state_layer_range must be positive.")
        return 1, layer_range

    layer_range = str(layer_range).strip()
    if not layer_range:
        raise ValueError("linear_state_layer_range cannot be empty.")
    try:
        if ":" not in layer_range:
            end = int(layer_range)
            if end < 1:
                raise ValueError
            return 1, end

        if layer_range.count(":") != 1:
            raise ValueError("linear_state_layer_range must be positive.")
        start_text, end_text = layer_range.split(":", 1)
        start = int(start_text) if start_text else 1
        end = int(end_text) if end_text else None
        if start < 1 or (end is not None and end < start):
            raise ValueError
        return start, end
    except ValueError as exc:
        raise ValueError(
            "linear_state_layer_range must be 'all', 'N', or 'start:end' "
            "with 1 <= start <= end."
        ) from exc


def _get_current_linear_state_scores(self, past_key_values, seq_len):
    if past_key_values is None:
        return None

    layer_types = getattr(self.config, "layer_types", None)
    if layer_types is None:
        return None

    layer_start, layer_end = _parse_linear_state_layer_range(
        self.config.method_config.get("linear_state_layer_range", "all")
    )
    linear_state_scores = []
    prev_layer_idx = self.layer_idx - 1
    linear_layer_offset = 0
    while prev_layer_idx >= 0 and layer_types[prev_layer_idx] == "linear_attention":
        linear_layer_offset += 1
        if layer_end is not None and linear_layer_offset > layer_end:
            break
        prev_cache = past_key_values.layers[prev_layer_idx]
        if linear_layer_offset >= layer_start:
            scores = getattr(prev_cache, "linear_state_scores", None)
            if scores is not None and scores.shape[-1] >= seq_len:
                linear_state_scores.append(scores[:, :, -seq_len:])
        prev_layer_idx -= 1

    if not linear_state_scores:
        return None

    layer_reduce = self.config.method_config.get("linear_state_layer_reduce", "mean")
    stacked_scores = torch.stack(linear_state_scores, dim=0)
    if layer_reduce == "mean":
        scores = stacked_scores.mean(dim=0)
    elif layer_reduce == "max":
        scores = stacked_scores.max(dim=0).values
    else:
        raise ValueError(f"Unsupported linear_state_layer_reduce: {layer_reduce}")

    return _align_linear_state_heads(scores, self.config.num_key_value_heads)


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

        requires_linear_state_scores = getattr(
            self.kv_cluster,
            "requires_linear_state_scores",
            False,
        )
        overlap_uses_linear_state = (
            getattr(self.kv_cluster, "record_gate_overlap", False)
            and self.config.method_config.get("use_linear_state", True)
        )
        needs_linear_state_cache = (
            requires_linear_state_scores
            or overlap_uses_linear_state
        )
        if needs_linear_state_cache:
            current_linear_state_scores = _get_current_linear_state_scores(
                self,
                past_key_values,
                query_states.shape[-2],
            )
            if current_linear_state_scores is not None:
                if (
                    not hasattr(layer_cache, "linear_state_cache")
                    or layer_cache.linear_state_cache is None
                ):
                    layer_cache.linear_state_cache = current_linear_state_scores
                else:
                    layer_cache.linear_state_cache = torch.cat(
                        (layer_cache.linear_state_cache, current_linear_state_scores),
                        dim=2,
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
            if needs_linear_state_cache:
                update_kwargs["linear_state_scores"] = getattr(
                    layer_cache,
                    "linear_state_cache",
                    None,
                )
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
                    if (
                        needs_linear_state_cache
                        and kept_indices is not None
                        and getattr(layer_cache, "linear_state_cache", None) is not None
                    ):
                        layer_cache.linear_state_cache = _compress_linear_state_cache(
                            layer_cache.linear_state_cache,
                            kept_indices,
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
            if needs_linear_state_cache:
                update_kwargs["linear_state_scores"] = getattr(
                    layer_cache,
                    "linear_state_cache",
                    None,
                )
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
                    if (
                        needs_linear_state_cache
                        and kept_indices is not None
                        and getattr(layer_cache, "linear_state_cache", None) is not None
                    ):
                        layer_cache.linear_state_cache = _compress_linear_state_cache(
                            layer_cache.linear_state_cache,
                            kept_indices,
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


def Qwen3_5GatedDeltaNet_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params: Cache | None = None,
    attention_mask: torch.Tensor | None = None,
):
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

    batch_size, seq_len, _ = hidden_states.shape

    use_precomputed_states = (
        cache_params is not None
        and cache_params.has_previous_state(self.layer_idx)
        and seq_len == 1
    )

    if use_precomputed_states:
        conv_state = cache_params.layers[self.layer_idx].conv_states
        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if use_precomputed_states:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        if cache_params is not None:
            conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [
            self.key_dim,
            self.key_dim,
            self.value_dim,
        ],
        dim=-1,
    )

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if cache_params is not None:
        key_normed = l2norm(key, dim=-1, eps=1e-6)
        future_g = torch.flip(
            torch.cumsum(torch.flip(g.float(), dims=[1]), dim=1),
            dims=[1],
        ) - g.float()
        write_score = (
            key_normed.float().norm(p=2, dim=-1)
            * value.float().norm(p=2, dim=-1)
            * beta.float()
            * future_g.exp()
        )
        if attention_mask is not None and attention_mask.ndim == 2:
            write_score = write_score * attention_mask[:, -seq_len:, None].to(write_score.dtype)
        cache_params.layers[self.layer_idx].linear_state_scores = (
            write_score.transpose(1, 2).detach()
        )

    if not use_precomputed_states:
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)
    return output


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
