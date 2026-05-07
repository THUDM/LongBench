#!/bin/bash
# Needle-in-a-Haystack on Qwen3.5-9B 
# Tests at budget=128L and budget=1024L

MODEL_PATH="Qwen/Qwen3.5-9B"  # e.g., meta-llama/Llama-3.1-8B-Instruct
ATTN="flash_attention_2"

EXPERIMENTS=(
    "FullKV 128 Qwen3.5"
)

CUDA_DEVICES=("0" "1" "2" "3")
NUM_GPUS=${#CUDA_DEVICES[@]}
IDX=0
PIDS=()

cleanup() {
    trap - INT TERM
    if ((${#PIDS[@]} > 0)); then
        echo
        echo "Interrupted. Stopping running Needle Qwen3.5 jobs..."
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
    read -r method capacity provider <<< "$exp"
    gpu=${CUDA_DEVICES[$((IDX % NUM_GPUS))]}

    compression_args=()
    version_args=(--model_version "Qwen3.5_FullKV_${capacity}")
    if [[ "${method}" != "FullKV" ]]; then
        compression_args=(
            --compression
            --compression_mode "${method}"
            --compression_budget "${capacity}"
        )
        version_args=(--model_version "Qwen3.5")
    fi

    export CUDA_VISIBLE_DEVICES=$gpu
    python -u run_niah.py \
        --s_len 1000 --e_len 32001 \
        --model_provider "${provider}" \
        --model_name "${MODEL_PATH}" \
        --attn_implementation "${ATTN}" \
        --step 1000 \
        "${version_args[@]}" \
        "${compression_args[@]}" &
    PIDS+=("$!")

    IDX=$((IDX + 1))
    if (( IDX % NUM_GPUS == 0 )); then
        wait_for_batch
    fi
done
wait_for_batch
echo "All Needle Qwen3.5 experiments completed."
