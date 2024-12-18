#!/bin/bash
chunk_size=200
work_dir="../../LongBench" # dir for storing data

source_dir="${work_dir}/data" # source LongBench dir
chunk_dir="C${chunk_size}"
split_dir="${work_dir}/${chunk_dir}/split"
embed_dir="${work_dir}/${chunk_dir}/embed"
retrieved_dir="${work_dir}/${chunk_dir}/output"
python LB2mC.py \
    --chunk_size ${chunk_size} \
    --output_folder  ${split_dir}\
    --input_folder ${source_dir}
folder_names=()
# Traverse all subfolders under `split` dir
for folder in "$split_dir"/*; do
    if [ -d "$folder" ]; then
        # get the name of subfolder
        folder_name=$(basename "$folder")
        # concat
        folder_path="$split_dir/$folder_name"
        echo "$folder_path"
        folder_names+=("$folder_name")
    fi
done

# Traverse all subfolders under `split` dir
for folder in "${folder_names[@]}"; do
    file_paths=()
  # Traverse all files in a subfolder
    for file in "$split_dir"/"$folder"/*.tsv; do
      if [ -f "$file" ]; then
        fileName=$(basename "${file%.*}")
        file_paths+=("${split_dir}/${folder}/${fileName}.tsv")
      fi
    done
    # Converts an array to a ' ' separated string
    files_str=$(IFS=' '; echo "${file_paths[*]}")
    # generate embeddings
    python ./contriever/generate_passage_embeddings.py \
            --model_name_or_path ./contriever/mcontriever \
            --output_dir ${embed_dir}/${folder}  \
            --psgs_list  $files_str\
            --shard_id 0 --num_shards 1 \
            --lowercase --normalize_text 

    # generate results of retrieval
    tsv_files=("$split_dir/$folder"/*.tsv)
    # concurrent execution
    group_size=5

    for ((start=0; start<${#tsv_files[@]}; start+=group_size)); do
      end=$((start + group_size - 1))
      echo "Index Range：$start ~ $end"
      current_group=("${tsv_files[@]:start:group_size}")

      for ((index=0; index<${#current_group[@]}; index+=1)); do
        file=${current_group[index]}
        fileName=$(basename "${file%.*}")
        python ./contriever/passage_retrieval.py \
        --model_name_or_path ./contriever/mcontriever \
        --passages ${split_dir}/${folder}/${fileName}.tsv \
        --passages_embeddings ${embed_dir}/${folder}/${fileName} \
        --data ${split_dir}/${folder}/${fileName}.jsonl \
        --output_dir ${retrieved_dir}/${folder} \
        --lowercase --normalize_text \
        --device "cuda" \
        &
        # --device "cuda:$(expr 4 + $index % 4)" \
      done
      wait
    done

    python merge_output.py \
    --input_folder  "${retrieved_dir}/${folder}" \
    --output_file "${work_dir}/${chunk_dir}/mc2LB/${folder}.jsonl" \
    --input_dataFile "${source_dir}/${folder}.jsonl" \
    --output_dataFile "${work_dir}/${chunk_dir}/data/${folder}.jsonl"
done

cp ../LongBench.py "${work_dir}/${chunk_dir}"