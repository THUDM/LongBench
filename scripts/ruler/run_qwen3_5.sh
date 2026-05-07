#!/bin/bash
# RULER benchmark on Qwen3.5-9B
# Tests across context lengths 4K-128K, budget=1024L

MODEL_PATH="Qwen/Qwen3.5-9B"  # e.g., meta-llama/Llama-3.1-8B-Instruct
SAVE_DIR="output_dir/results_ruler"
ATTN="flash_attention_2"
MODEL_MAXLEN=200000

EXPERIMENTS=(
    "FullKV 0"
)

CUDA_DEVICES=("0" "1" "2" "3")
NUM_GPUS=${#CUDA_DEVICES[@]}
IDX=0
PIDS=()

cleanup() {
    trap - INT TERM
    if ((${#PIDS[@]} > 0)); then
        echo
        echo "Interrupted. Stopping running RULER Qwen3.5 jobs..."
        kill -TERM "${PIDS[@]}" 2>/dev/null || true
        wait "${PIDS[@]}" 2>/dev/null || true
    fi
    exit 130
}

wait_for_batch() {
    if ((${#PIDS[@]} == 0)); then
        return
    fi
    wait "${PIDS[@]}"
    PIDS=()
}

trap cleanup INT TERM

for exp in "${EXPERIMENTS[@]}"; do
    read -r method budget <<< "$exp"
    gpu=${CUDA_DEVICES[$((IDX % NUM_GPUS))]}

    compression_args=()
    if [[ "${method}" != "FullKV" ]]; then
        compression_args=(
            --compression
            --compression_mode "${method}"
            --compression_budget "${budget}"
        )
    fi

    export CUDA_VISIBLE_DEVICES=$gpu
    python3 -u run_ruler.py \
        --model_path "${MODEL_PATH}" \
        --attn_implementation "${ATTN}" \
        --save_dir "${SAVE_DIR}" \
        --model_maxlen "${MODEL_MAXLEN}" \
        "${compression_args[@]}" &
    PIDS+=("$!")

    IDX=$((IDX + 1))
    if (( IDX % NUM_GPUS == 0 )); then
        wait_for_batch
    fi
done
wait_for_batch
echo "All RULER Qwen3.5 experiments completed."
