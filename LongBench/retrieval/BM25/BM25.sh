#!/bin/bash
chunk_size=500
work_dir="../../LongBench" # dir for storing data

source_dir="${work_dir}/data" # source LongBench dir
dest_dir=""${work_dir}/B${chunk_size}/data""

file_names=()
allowed_files=("multifieldqa_en.jsonl" "qasper.jsonl" "2wikimqa.jsonl" "dureader.jsonl" "hotpotqa.jsonl" "narrativeqa.jsonl" "musique.jsonl" "multifieldqa_zh.jsonl")
# store all jsonl files
while IFS= read -r -d '' file; do
    base_name=$(basename "$file")
    # Check if the file name is in the allowed_files list
    if [[ " ${allowed_files[@]} " =~ " ${base_name} " ]]; then
        file_names+=("$base_name")
    fi
done < <(find "$source_dir" -type f -name "*.jsonl" -print0)

# concurrent execution
group_size=3

for ((start=0; start<${#file_names[@]}; start+=group_size)); do
    end=$((start + group_size - 1))
    echo "Index Range：$start ~ $end"
    current_group=("${file_names[@]:start:group_size}")
    for file in "${current_group[@]}"; do
        fileName=$(basename "${file}")
        python generate_BM25.py \
        --file_name $fileName \
        --source_dir $source_dir \
        --dest_dir $dest_dir \
        --chunk_size $chunk_size \
        &
    done
    wait
done

cp ../LongBench.py "${work_dir}/B${chunk_size}"