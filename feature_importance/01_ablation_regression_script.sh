#!/bin/bash

# Generate a random seed
seed=$(shuf -i 1-10000 -n 1)

# Replace XXX with the generated seed
command="01_run_ablation_regression.py --nreps 1 --config mdi_local.real_data_regression --split_seed $seed --ignore_cache --create_rmd --result_name Diabetes_regression_parallel"

# Execute the command
python $command