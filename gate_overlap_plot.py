import os
import traceback

from attn_heatmap import sanitize_slug


def iter_gate_overlap_clusters(model):
    model_body = getattr(model, "model", model)
    layers = getattr(model_body, "layers", None)
    if layers is None and hasattr(model_body, "language_model"):
        layers = getattr(model_body.language_model, "layers", None)
    if layers is None:
        return

    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        kv_cluster = getattr(self_attn, "kv_cluster", None)
        if kv_cluster is not None and getattr(kv_cluster, "record_gate_overlap", False):
            yield kv_cluster


def clear_gate_overlap_records(model):
    for kv_cluster in iter_gate_overlap_clusters(model):
        for attr_name in (
            "method_topk_indices",
            "gate_topk_indices",
            "gate_method_overlap",
            "snapkv_topk_indices",
            "gate_snapkv_overlap",
        ):
            records = getattr(kv_cluster, attr_name, None)
            if isinstance(records, list):
                records.clear()


def build_gate_overlap_output_path(args, out_file, item, sample_index):
    run_name = os.path.splitext(os.path.basename(out_file))[0]
    run_dir = os.path.join(args.gate_overlap_dir, sanitize_slug(run_name, fallback="run"))
    os.makedirs(run_dir, exist_ok=True)

    file_name = f"sample_{sample_index:06d}_gate_method_overlap.png"
    return os.path.join(run_dir, file_name)


def save_gate_overlap_plot(model, args, out_file, item, sample_index):
    if not args.gate_overlap_mode:
        return None
    from models.compression.utils import plot_gate_method_overlap

    output_path = build_gate_overlap_output_path(args, out_file, item, sample_index)
    try:
        result = plot_gate_method_overlap(
            model,
            output_path,
            interpolation=args.gate_overlap_interpolation,
        )
    except ValueError as exc:
        sample_id = item.get("_id", f"sample_{sample_index:06d}")
        print(f"Skipping gate/method overlap plot for {sample_id}: {exc}")
        return None
    except Exception as exc:
        sample_id = item.get("_id", f"sample_{sample_index:06d}")
        print(f"Could not save gate/method overlap plot for {sample_id}: {exc}")
        traceback.print_exc()
        return None

    print(
        "Saved gate/method overlap plot to "
        f"{result['output_path']} with matrix shape "
        f"{list(result['overlap_matrix'].shape)}."
    )
    return result["output_path"]
