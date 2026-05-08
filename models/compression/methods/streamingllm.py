import torch
from .gate_overlap import init_gate_overlap_recorder, record_gate_topk_overlap


class StreamingLLM:
    def __init__(
        self,
        budget=128,
        first_tokens=4,
        record_gate_overlap=False,
        layer_idx=None,
        model_config=None,
        model_type=None,
        mode=None,
        **kwargs,
    ):
        assert budget - first_tokens > 0, "budget must be greater than first_tokens"
        self.budget = budget
        self.first_tokens = first_tokens
        self.layer_idx = layer_idx
        self.model_config = model_config
        self.model_type = model_type
        self.mode = mode
        init_gate_overlap_recorder(self, "streamingllm", record_gate_overlap)

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        gate_states=None,
        linear_state_scores=None,
        is_prefill=False,
    ):
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len < self.budget:
            return key_states, value_states
        else:
            local_window_size = self.budget - self.first_tokens
            first_indices = torch.arange(
                self.first_tokens,
                device=key_states.device,
            )
            recent_indices = torch.arange(
                kv_cache_len - local_window_size,
                kv_cache_len,
                device=key_states.device,
            )
            indices = torch.cat([first_indices, recent_indices], dim=0)
            indices = indices.view(1, 1, -1).expand(
                key_states.shape[0],
                key_states.shape[1],
                -1,
            )
            record_gate_topk_overlap(
                self,
                method_indices=indices,
                gate_states=gate_states,
                kv_cache_len=kv_cache_len,
                is_prefill=is_prefill,
                linear_state_scores=linear_state_scores,
            )
            # only select the first self.first_tokens tokens and the last local_window_size tokens
            key_states = torch.cat(
                [
                    key_states[:, :, : self.first_tokens],
                    key_states[:, :, -local_window_size:],
                ],
                dim=2,
            )
            value_states = torch.cat(
                [
                    value_states[:, :, : self.first_tokens],
                    value_states[:, :, -local_window_size:],
                ],
                dim=2,
            )
            return key_states, value_states
