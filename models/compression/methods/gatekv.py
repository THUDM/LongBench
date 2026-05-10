import torch

class GateKV:
    requires_gate_states = True

    def __init__(
        self,
        budget=128,
        window_size=8,
        first_tokens=4,
        use_linear_state=True,
        linear_state_weight=0.3,
        linear_state_required=False,
        linear_state_norm="rank",
        record_kept_token_indices=False,
        record_gate_overlap=False,
        layer_idx=None,
        model_config=None,
        model_type=None,
        mode=None,
        **kwargs,
    ):
        assert budget - first_tokens - window_size > 0, (
            "budget must be greater than first_tokens + window_size"
        )
        if not 0.0 <= linear_state_weight <= 1.0:
            raise ValueError("linear_state_weight must be in [0, 1].")
        self.budget = budget
        self.window_size = window_size
        self.first_tokens = first_tokens
        self.use_linear_state = use_linear_state
        self.linear_state_weight = linear_state_weight
        self.linear_state_required = linear_state_required
        self.linear_state_norm = linear_state_norm
        self.requires_linear_state_scores = use_linear_state and linear_state_weight > 0

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

    def _normalize_scores(self, scores):
        if self.linear_state_norm == "rank":
            if scores.shape[-1] == 1:
                return torch.ones_like(scores, dtype=torch.float32)
            ranks = scores.argsort(dim=-1).argsort(dim=-1).to(torch.float32)
            return ranks / (scores.shape[-1] - 1)
        if self.linear_state_norm == "minmax":
            scores = scores.to(torch.float32)
            min_scores = scores.min(dim=-1, keepdim=True).values
            max_scores = scores.max(dim=-1, keepdim=True).values
            return (scores - min_scores) / (max_scores - min_scores).clamp_min(1e-6)
        if self.linear_state_norm == "none":
            return scores.to(torch.float32)
        raise ValueError(f"Unsupported linear_state_norm: {self.linear_state_norm}")

    def _linear_state_scores(
        self,
        linear_state_scores,
        batch_size,
        num_kv_heads,
        kv_cache_len,
    ):
        if linear_state_scores is None:
            if self.linear_state_required:
                raise ValueError("GateKV linear state enhancement requires linear_state_scores.")
            return None
        if linear_state_scores.ndim != 3:
            raise ValueError(
                "linear_state_scores must have shape "
                "(batch, num_key_value_heads, kv_cache_len)."
            )
        if linear_state_scores.shape[0] != batch_size:
            raise ValueError(
                f"linear state batch size {linear_state_scores.shape[0]} does not "
                f"match KV batch size {batch_size}."
            )
        if linear_state_scores.shape[1] != num_kv_heads:
            raise ValueError(
                f"linear state heads {linear_state_scores.shape[1]} do not match "
                f"KV heads {num_kv_heads}."
            )
        if linear_state_scores.shape[2] < kv_cache_len:
            if self.linear_state_required:
                raise ValueError(
                    f"linear state sequence length {linear_state_scores.shape[2]} "
                    f"is shorter than KV cache length {kv_cache_len}."
                )
            return None
        return linear_state_scores[:, :, :kv_cache_len]

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        gate_states=None,
        linear_state_scores=None,
        is_prefill=False,
    ):
        head_dim = key_states.shape[-1]
        batch_size = key_states.shape[0]
        num_kv_heads = key_states.shape[1]
        kv_cache_len = key_states.shape[-2]
        self.last_kept_indices = None

        if kv_cache_len <= self.budget:
            return key_states, value_states

        middle_start = self.first_tokens
        middle_end = kv_cache_len - self.window_size
        middle_budget = self.budget - self.first_tokens - self.window_size

        gate_scores = self._gate_scores(
            gate_states,
            batch_size,
            num_kv_heads,
            kv_cache_len,
        )
        gate_scores = gate_scores[:, :, middle_start:middle_end]

        state_scores = None
        if self.use_linear_state and self.linear_state_weight > 0:
            state_scores = self._linear_state_scores(
                linear_state_scores,
                batch_size,
                num_kv_heads,
                kv_cache_len,
            )
            if state_scores is not None:
                state_scores = state_scores[:, :, middle_start:middle_end]

        if state_scores is not None:
            gate_weight = 1.0 - self.linear_state_weight
            gate_scores = (
                gate_weight * self._normalize_scores(gate_scores)
                + self.linear_state_weight * self._normalize_scores(state_scores)
            ).to(gate_scores.dtype)

        # shape: (bsz, num_kv_heads, budget - first_tokens - window_size)
        indices = gate_scores.topk(middle_budget, dim=-1).indices
        middle_indices = indices + middle_start

        first_indices = torch.arange(
            self.first_tokens,
            device=key_states.device,
        ).view(1, 1, -1).expand(batch_size, num_kv_heads, -1)
        recent_window_indices = torch.arange(
            kv_cache_len - self.window_size,
            kv_cache_len,
            device=key_states.device,
        ).view(1, 1, -1).expand(batch_size, num_kv_heads, -1)
        kept_indices = torch.cat(
            [first_indices, middle_indices, recent_window_indices],
            dim=-1,
        )
        self.last_kept_indices = kept_indices.detach()

        #####################################################
        ###### Store evicted token indices start ############
        #####################################################
        # shape: (num_kv_heads, budget)
        if self.record_kept_token_indices:
            cur_indices = kept_indices.clone().squeeze(0).to("cpu")

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
        k_first = key_states[:, :, : self.first_tokens, :]
        v_first = value_states[:, :, : self.first_tokens, :]
        k_middle = key_states[:, :, middle_start:middle_end, :].gather(
            dim=2,
            index=indices,
        )
        v_middle = value_states[:, :, middle_start:middle_end, :].gather(
            dim=2,
            index=indices,
        )
        k_cur = key_states[:, :, -self.window_size :, :]
        v_cur = value_states[:, :, -self.window_size :, :]
        key_states = torch.cat([k_first, k_middle, k_cur], dim=2)
        value_states = torch.cat([v_first, v_middle, v_cur], dim=2)

        return key_states, value_states
