import torch

from .gate_scoring import (
    align_gate_scores,
    combine_gate_with_linear_state,
    select_topk_indices,
    validate_linear_state_scores,
)


def init_gate_overlap_recorder(method, method_name, record_gate_overlap=False):
    method.record_gate_overlap = record_gate_overlap
    if not method.record_gate_overlap:
        return

    method.gate_overlap_method_name = method_name
    method.method_topk_indices = []
    method.gate_topk_indices = []
    method.gate_method_overlap = []
    method.method_pre_topk_scores = []
    method.gate_pre_topk_scores = []
    method.pre_topk_score_positions = []

    # Backward-compatible aliases for the original SnapKV-only names.
    method.snapkv_topk_indices = method.method_topk_indices
    method.gate_snapkv_overlap = method.gate_method_overlap


def _expand_candidate_indices(candidate_indices, batch_size, num_kv_heads, device):
    if candidate_indices.ndim == 1:
        candidate_indices = candidate_indices.view(1, 1, -1)
    elif candidate_indices.ndim == 2:
        candidate_indices = candidate_indices.unsqueeze(0)
    elif candidate_indices.ndim != 3:
        raise ValueError(
            "candidate_indices must have shape (candidate_len,), "
            "(num_kv_heads, candidate_len), or "
            "(batch, num_kv_heads, candidate_len)."
        )

    candidate_indices = candidate_indices.to(device=device, dtype=torch.long)
    if candidate_indices.shape[0] == 1 and batch_size != 1:
        candidate_indices = candidate_indices.expand(batch_size, -1, -1)
    if candidate_indices.shape[1] == 1 and num_kv_heads != 1:
        candidate_indices = candidate_indices.expand(-1, num_kv_heads, -1)

    if candidate_indices.shape[:2] != (batch_size, num_kv_heads):
        raise ValueError(
            f"candidate_indices shape {tuple(candidate_indices.shape)} cannot "
            f"broadcast to batch={batch_size}, kv_heads={num_kv_heads}."
        )
    return candidate_indices


def _method_config(method):
    model_config = getattr(method, "model_config", None)
    return getattr(model_config, "method_config", {}) or {}


def _enhance_gate_scores(method, gate_scores, linear_state_scores, kv_cache_len):
    config = _method_config(method)
    use_linear_state = config.get(
        "use_linear_state",
        getattr(method, "use_linear_state", True),
    )
    linear_state_weight = config.get(
        "linear_state_weight",
        getattr(method, "linear_state_weight", 0.3),
    )
    linear_state_required = config.get(
        "linear_state_required",
        getattr(method, "linear_state_required", False),
    )
    linear_state_norm = config.get(
        "linear_state_norm",
        getattr(method, "linear_state_norm", "rank"),
    )

    if not use_linear_state or linear_state_weight <= 0:
        return gate_scores
    linear_state_scores = validate_linear_state_scores(
        linear_state_scores,
        gate_scores.shape[0],
        gate_scores.shape[1],
        kv_cache_len,
        required=linear_state_required,
        missing_message="linear_state_scores are required for enhanced gate overlap.",
    )
    return combine_gate_with_linear_state(
        gate_scores,
        linear_state_scores,
        linear_state_weight,
        linear_state_norm,
    )


def record_gate_topk_overlap(
    method,
    method_indices,
    gate_states,
    kv_cache_len,
    is_prefill,
    candidate_indices=None,
    method_scores=None,
    linear_state_scores=None,
):
    if not getattr(method, "record_gate_overlap", False) or not is_prefill:
        return
    if gate_states is None:
        return
    if gate_states.ndim != 4:
        raise ValueError(
            "gate_states must have shape "
            "(batch, seq_len, num_attention_heads, head_dim)."
        )
    if method_indices.ndim != 3:
        raise ValueError(
            "method_indices must have shape (batch, num_kv_heads, k)."
        )

    batch_size, num_kv_heads, k = method_indices.shape
    if k <= 0:
        return
    if gate_states.shape[0] != batch_size:
        raise ValueError(
            f"gate batch size {gate_states.shape[0]} does not match "
            f"method index batch size {batch_size}."
        )
    if gate_states.shape[1] < kv_cache_len:
        raise ValueError(
            f"gate sequence length {gate_states.shape[1]} is shorter than "
            f"KV cache length {kv_cache_len}."
        )

    if candidate_indices is None:
        candidate_indices = torch.arange(kv_cache_len, device=method_indices.device)
    device = gate_states.device
    method_indices = method_indices.to(device=device, dtype=torch.long)
    candidate_indices = _expand_candidate_indices(
        candidate_indices,
        batch_size,
        num_kv_heads,
        device,
    )

    candidate_len = candidate_indices.shape[-1]
    if candidate_len < k:
        raise ValueError(
            f"candidate length {candidate_len} is smaller than topk size {k}."
        )

    method_abs_indices = torch.gather(candidate_indices, dim=-1, index=method_indices)

    if method_scores is not None:
        method_scores = method_scores.to(device)
        if method_scores.shape != candidate_indices.shape:
            raise ValueError(
                f"method_scores shape {tuple(method_scores.shape)} must match "
                f"candidate_indices shape {tuple(candidate_indices.shape)}."
            )

    gate_scores = align_gate_scores(
        gate_states,
        batch_size,
        num_kv_heads,
        kv_cache_len,
    )
    gate_scores = _enhance_gate_scores(
        method,
        gate_scores,
        linear_state_scores,
        kv_cache_len,
    )
    gate_scores = gate_scores.to(device)
    gate_scores = torch.gather(gate_scores, dim=-1, index=candidate_indices)
    gate_scores_for_plot = gate_scores

    gate_local_indices = select_topk_indices(gate_scores, k, dim=-1, softmax=True)
    gate_abs_indices = torch.gather(candidate_indices, dim=-1, index=gate_local_indices)
    overlap = (
        (method_abs_indices.unsqueeze(-1) == gate_abs_indices.unsqueeze(-2))
        .any(dim=-1)
        .sum(dim=-1)
        .to(dtype=torch.float32)
        / float(k)
    )

    method.method_topk_indices.append(method_abs_indices.detach().cpu())
    method.gate_topk_indices.append(gate_abs_indices.detach().cpu())
    method.gate_method_overlap.append(overlap.detach().cpu())
    if method_scores is not None:
        method.method_pre_topk_scores.append(method_scores.detach().cpu())
        method.gate_pre_topk_scores.append(gate_scores_for_plot.detach().cpu())
        method.pre_topk_score_positions.append(candidate_indices.detach().cpu())
