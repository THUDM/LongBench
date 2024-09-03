MODEL_PATH=/cb/cold/checkpoints/llms/HF/Meta-Llama-3-8B-Instruct #Phi-3-mini-128k-instruct
IFS='/' read -r -a array <<< $MODEL_PATH
MODEL_NAME=${array[-1]}
DATASET=humaneval
SEQ_LEN=7500
KV_METHOD=full

~/miniconda3/envs/longbench/bin/python pred.py --model $MODEL_PATH --e --max_seq_len $SEQ_LEN --kv_method $KV_METHOD --backend hf #--load_in_8bit
# ~/miniconda3/envs/longbench/bin/python eval.py --model Meta-Llama-3-8B-Instruct_7500_kv_cache_full_backend_vllm --e #--max_seq_len 3500