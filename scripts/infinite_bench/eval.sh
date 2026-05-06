#!/bin/bash
# Evaluate InfiniteBench results
# Usage: bash eval.sh <results_dir>

results_dir=$1
python3 eval_infinite.py --output_dir ${results_dir} --task all