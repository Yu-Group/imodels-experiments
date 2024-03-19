#!/bin/bash

# Generate a random seed
seed=$(shuf -i 1-10000 -n 1)

# Replace XXX with the generated seed
command="01_run_ablation_classification.py --nreps 1 --config mdi_local.real_data_classification --split_seed $seed --ignore_cache --create_rmd --result_name Diabetes_classification_parallel"

# Execute the command
python $command