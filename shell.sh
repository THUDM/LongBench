###### attention heatmap visualization for qwen3.5-9b

CUDA_VISIBLE_DEVICES=0,1 python run_longbench.py --model qwen3.5-9b --attn_heatmap_mode --num_samples 3 --attn_max_prefill_tokens 2500 --domain "single-document QA" --model_maxlen 2000

python attn_viewer/server.py --root results/attn_heatmaps

###### gate indices overlap visualization for qwen3.5-9b

CUDA_VISIBLE_DEVICES=0,1 python run_longbench.py --model qwen3.5-9b --gate_overlap_mode --num_samples 3 --domain "single-document QA" --compression --compression_mode "snapkv" --compression_budget 1024 --model_maxlen 10000 --use_linear_state --linear_state_weight 1 --linear_state_norm rank --linear_state_layer_range 1 --linear_state_score_type "write_norm"
##### efficiency benchmark for qwen3.5-9b

for input_len in 10000 20000 30000 40000 50000 60000; do
    echo "Running benchmark with input_len=${input_len}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_efficiency.py \
        --model_path Qwen/Qwen3.5-9B \
        --batch_size 1 \
        --input_len "${input_len}" \
        --output_len 128 \
        --num_warmups 1 \
        --num_runs 2
done

for input_len in 10000 20000 30000 40000 50000 60000; do
    echo "Running benchmark with input_len=${input_len}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_efficiency.py \
        --model_path Qwen/Qwen3.5-9B \
        --batch_size 1 \
        --input_len "${input_len}" \
        --output_len 128 \
        --compression \
        --compression_mode "gatekv" \
        --compression_budget 1024 \
        --num_warmups 1 \
        --num_runs 2
done

for input_len in 10000 20000 30000 40000 50000 60000; do
    echo "Running benchmark with input_len=${input_len}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_efficiency.py \
        --model_path Qwen/Qwen3.5-9B \
        --batch_size 1 \
        --input_len "${input_len}" \
        --output_len 128 \
        --compression \
        --compression_mode "snapkv" \
        --compression_budget 1024 \
        --num_warmups 1 \
        --num_runs 2
done

##### compression benchmark for qwen3.5-9b

for domain in "Code Repository Understanding" "Long In-context Learning"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_longbench.py \
        --model qwen3.5-9b \
        --domain "$domain" \
        --model_maxlen 50000

    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_longbench.py \
        --model qwen3.5-9b \
        --domain "$domain" \
        --compression \
        --compression_mode "snapkv" \
        --compression_budget 1024 \
        --model_maxlen 50000 

    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_longbench.py \
        --model qwen3.5-9b \
        --domain "$domain" \
        --compression \
        --compression_mode "gatekv" \
        --compression_budget 1024 \
        --model_maxlen 50000 --use_linear_state --linear_state_weight 1 --linear_state_norm rank --linear_state_layer_range 1 --linear_state_score_type "write_norm"
done
