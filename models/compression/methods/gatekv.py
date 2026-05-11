import torch

from .gate_scoring import (
    align_gate_scores,
    combine_gate_with_linear_state,
    select_topk_indices,
    validate_linear_state_scores,
)


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

        gate_scores = align_gate_scores(
            gate_states,
            batch_size,
            num_kv_heads,
            kv_cache_len,
            missing_message="GateKV requires gate_states from self_attn.gate.",
        )
        gate_scores = gate_scores[:, :, middle_start:middle_end]

        state_scores = None
        if self.use_linear_state and self.linear_state_weight > 0:
            state_scores = validate_linear_state_scores(
                linear_state_scores,
                batch_size,
                num_kv_heads,
                kv_cache_len,
                required=self.linear_state_required,
                missing_message="GateKV linear state enhancement requires linear_state_scores.",
            )
            if state_scores is not None:
                state_scores = state_scores[:, :, middle_start:middle_end]

        gate_scores = combine_gate_with_linear_state(
            gate_scores,
            state_scores,
            self.linear_state_weight,
            self.linear_state_norm,
        )

        # shape: (bsz, num_kv_heads, budget - first_tokens - window_size)
        indices = select_topk_indices(gate_scores, middle_budget, dim=-1)
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
