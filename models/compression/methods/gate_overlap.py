import torch
import torch.nn.functional as F


def init_gate_overlap_recorder(method, method_name, record_gate_overlap=False):
    method.record_gate_overlap = record_gate_overlap
    if not method.record_gate_overlap:
        return

    method.gate_overlap_method_name = method_name
    method.method_topk_indices = []
    method.gate_topk_indices = []
    method.gate_method_overlap = []

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


def record_gate_topk_overlap(
    method,
    method_indices,
    gate_states,
    kv_cache_len,
    is_prefill,
    candidate_indices=None,
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

    gate_scores = gate_states[:, :kv_cache_len, :, :].norm(p=1, dim=-1).transpose(1, 2)
    gate_candidate_indices = candidate_indices[:, :1, :].expand(
        batch_size,
        gate_scores.shape[1],
        candidate_len,
    )
    gate_scores = torch.gather(gate_scores, dim=-1, index=gate_candidate_indices)
    gate_scores = F.softmax(gate_scores, dim=-1, dtype=torch.float32).to(
        gate_states.dtype
    )

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
            candidate_len,
        ).max(dim=2).values

    gate_local_indices = gate_scores.topk(k, dim=-1).indices
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
