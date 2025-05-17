#!/bin/bash

slurm_script="00_regression_selection_vary_sample_size_script.sh"

for data_name in "openml_361260" "openml_361622"; do # "openml_361242" "openml_361253" "openml_361254" "openml_361259" "openml_361260" "openml_361622"
    for split_seed in {1..4}; do
        for sample_seed in {1..5}; do
            sbatch $slurm_script $data_name $split_seed $sample_seed
            sleep 1
        done
    done
done