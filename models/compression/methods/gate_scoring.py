import torch
import torch.nn.functional as F


def normalize_scores(scores, norm):
    if norm == "rank":
        if scores.shape[-1] == 1:
            return torch.ones_like(scores, dtype=torch.float32)
        ranks = scores.argsort(dim=-1).argsort(dim=-1).to(torch.float32)
        return ranks / (scores.shape[-1] - 1)
    if norm == "minmax":
        scores = scores.to(torch.float32)
        min_scores = scores.min(dim=-1, keepdim=True).values
        max_scores = scores.max(dim=-1, keepdim=True).values
        return (scores - min_scores) / (max_scores - min_scores).clamp_min(1e-6)
    if norm == "none":
        return scores.to(torch.float32)
    raise ValueError(f"Unsupported linear_state_norm: {norm}")


def align_gate_scores(
    gate_states,
    batch_size,
    num_kv_heads,
    kv_cache_len,
    missing_message="gate_states are required.",
):
    if gate_states is None:
        raise ValueError(missing_message)
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

    gate_scores = gate_states[:, :kv_cache_len, :, :].norm(p=1, dim=-1).transpose(1, 2)

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


def validate_linear_state_scores(
    linear_state_scores,
    batch_size,
    num_kv_heads,
    kv_cache_len,
    required=False,
    missing_message="linear_state_scores are required.",
):
    if linear_state_scores is None:
        if required:
            raise ValueError(missing_message)
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
        if required:
            raise ValueError(
                f"linear state sequence length {linear_state_scores.shape[2]} "
                f"is shorter than KV cache length {kv_cache_len}."
            )
        return None
    return linear_state_scores[:, :, :kv_cache_len]


def combine_gate_with_linear_state(
    gate_scores,
    linear_state_scores,
    linear_state_weight,
    linear_state_norm,
):
    if linear_state_scores is None:
        return gate_scores
    if gate_scores.shape != linear_state_scores.shape:
        raise ValueError(
            f"linear_state_scores shape {tuple(linear_state_scores.shape)} "
            f"does not match gate scores shape {tuple(gate_scores.shape)}."
        )

    linear_state_scores = linear_state_scores.to(gate_scores.device)
    gate_weight = 1.0 - linear_state_weight
    return (
        gate_weight * normalize_scores(gate_scores, linear_state_norm)
        + linear_state_weight * normalize_scores(linear_state_scores, linear_state_norm)
    ).to(gate_scores.dtype)


def select_topk_indices(scores, k, dim=-1, softmax=False):
    if softmax:
        scores = F.softmax(scores, dim=dim, dtype=torch.float32).to(scores.dtype)
    return scores.topk(k, dim=dim).indices
