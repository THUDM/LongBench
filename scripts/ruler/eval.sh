#!/bin/bash
# Evaluate RULER results
# Usage: bash eval.sh <results_dir>

results_dir=$1
python3 eval_ruler.py --results_dir ${results_dir}