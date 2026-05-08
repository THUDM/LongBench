import torch

from .gate_overlap import init_gate_overlap_recorder


class GateKV:
    requires_gate_states = True

    def __init__(
        self,
        budget=128,
        window_size=8,
        record_kept_token_indices=False,
        record_gate_overlap=False,
        layer_idx=None,
        model_config=None,
        model_type=None,
        mode=None,
        **kwargs,
    ):
        assert budget > 0, "budget must be greater than 0"
        self.budget = budget
        self.window_size = window_size

        self.layer_idx = layer_idx
        self.model_config = model_config
        self.model_type = model_type
        self.mode = mode
        self.last_kept_indices = None

        # for recording kept token indices
        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []

        init_gate_overlap_recorder(self, "gatekv", record_gate_overlap)

    def _gate_scores(self, gate_states, batch_size, num_kv_heads, kv_cache_len):
        if gate_states is None:
            raise ValueError("GateKV requires gate_states from self_attn.gate.")
        if gate_states.ndim != 4:
            raise ValueError(
                "gate_states must have shape "
                "(batch, seq_len, num_attention_heads, head_dim)."
            )
        if gate_states.shape[0] != batch_size:
            raise ValueError(
                f"gate batch size {gate_states.shape[0]} does not match "
                f"KV batch size {batch_size}."
            )
        if gate_states.shape[1] < kv_cache_len:
            raise ValueError(
                f"gate sequence length {gate_states.shape[1]} is shorter than "
                f"KV cache length {kv_cache_len}."
            )

        gate_scores = gate_states[:, :kv_cache_len, :, :].norm(p=1, dim=-1)
        gate_scores = gate_scores.transpose(1, 2)

        num_attention_heads = gate_scores.shape[1]
        if num_attention_heads != num_kv_heads:
            if num_attention_heads % num_kv_heads != 0:
                raise ValueError(
                    f"Cannot align {num_attention_heads} attention heads to "
                    f"{num_kv_heads} KV heads."
                )
            query_group_size = num_attention_heads // num_kv_heads
            gate_scores = gate_scores.view(
                batch_size,
                num_kv_heads,
                query_group_size,
                kv_cache_len,
            ).max(dim=2).values

        return gate_scores

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        gate_states=None,
        is_prefill=False,
    ):
        head_dim = key_states.shape[-1]
        batch_size = key_states.shape[0]
        num_kv_heads = key_states.shape[1]
        kv_cache_len = key_states.shape[-2]
        self.last_kept_indices = None

        if kv_cache_len <= self.budget:
            return key_states, value_states

        gate_scores = self._gate_scores(
            gate_states,
            batch_size,
            num_kv_heads,
            kv_cache_len,
        )

        # shape: (bsz, num_kv_heads, budget)
        indices = gate_scores.topk(self.budget, dim=-1).indices
        self.last_kept_indices = indices.detach()

        #####################################################
        ###### Store evicted token indices start ############
        #####################################################
        # shape: (num_kv_heads, budget)
        if self.record_kept_token_indices:
            indices_cl = indices.clone().squeeze(0).to("cpu")
            cur_indices = indices_cl

            if self.evicted_token_num > 0:
                prev_indices = self.kept_token_indices[-1]
                mask = cur_indices < self.budget

                for i in range(cur_indices.shape[0]):
                    positions = torch.where(mask[i])[0]

                    # For each position, get the value and use it as an index into prev_indices
                    for pos in positions:
                        val = cur_indices[i, pos].item()
                        cur_indices[i, pos] = prev_indices[i, val]

                # For values >= self.budget, add the evicted token count
                cur_indices[~mask] += self.evicted_token_num

            self.kept_token_indices.append(cur_indices)
            self.evicted_token_num += kv_cache_len - self.budget
        ######################################################

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        key_states = key_states.gather(dim=2, index=indices)
        value_states = value_states.gather(dim=2, index=indices)
        return key_states, value_states
