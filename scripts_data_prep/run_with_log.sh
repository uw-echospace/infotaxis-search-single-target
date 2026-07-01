#!/bin/bash
script_path="$1"
script_name=$(basename "$script_path")

parent_dir=$(cd "$(dirname "$script_path")/.." && pwd)
log_dir="$parent_dir/data_prep_logs_2026"
mkdir -p "$log_dir"

logfile="$log_dir/${script_name%.*}.log"

source ~/miniforge3/bin/activate infotaxis_20260512_py312
python -u "$script_path" 2>&1 | tee "$logfile"