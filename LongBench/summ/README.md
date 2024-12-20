## Introduction
The `compress.py` script is designed to address the challenge of processing excessively long texts using large language models. It provides a solution to reduce both the runtime and inference costs of running these models on long texts. The script works by segmenting the lengthy input into smaller segments, processing them individually through a chosen model, and then concatenating the outputs to reconstruct the final summarized result.

## Applicability
The `compress.py` script is particularly useful for summarization tasks involving the following datasets:
- `qmsum.jsonl`
- `gov_report.jsonl`
- `vcsum.jsonl`
- `multinews.jsonl`

## Motivation
The motivation behind this script is to tackle the challenges posed by processing extremely lengthy texts using large language models. Running these models on overly long texts can result in excessive execution times and high inference costs. The script addresses this by dividing the input text into manageable segments, summarizing them using a model, and finally merging the summaries to reconstruct the original long text.

## Usage
1. Clone this repository.
2. Open the `compress.py` script in a text editor.
3. Replace the keys or paths for the models (`glm2`, `gpt-16k`, `Llama2`) with your desired paths.
4. Define the paths for your raw data folder (`raw_data_folder`), the new data folder (`new_data_folder`), and the folder for storing compressed context texts (`compressed_context_path`).
5. Uncomment the code related to `flash_attn` if you wish to accelerate Llama2's inference using flash attention. If not needed, you can comment out this section.
6. Save the changes.

## Features
- Supports checkpoint resumption, allowing you to resume processing from where it was last paused.
- Automatically loads files from checkpoints.
- Utilizes `flash_attn` to speed up Llama2 inference (can be commented out if not applicable).

## Example
Here's an example of how to use the script:
```shell
python compress.py --model glm2 --max_len 500
