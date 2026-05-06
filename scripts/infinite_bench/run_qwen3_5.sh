#!/bin/bash
# InfiniteBench on Qwen3.5-9B
# Budget=1024L

MODEL_PATH="<path_to_model>"  # e.g., meta-llama/Llama-3.1-8B-Instruct
SAVE_DIR="output_dir/results_infinite_bench"
ATTN="flash_attention_2"

EXPERIMENTS=(
    "FullKV 0"
    "snapkv 1024"
)

CUDA_DEVICES=("0" "1")
NUM_GPUS=${#CUDA_DEVICES[@]}
IDX=0

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
    python3 -u run_infinite.py \
        --model_path "${MODEL_PATH}" \
        --attn_implementation "${ATTN}" \
        --save_dir "${SAVE_DIR}" \
        --use_cache True \
        "${compression_args[@]}" &

    IDX=$((IDX + 1))
    if (( IDX % NUM_GPUS == 0 )); then
        wait
    fi
done
wait
echo "All InfiniteBench Qwen3.5 experiments completed."
