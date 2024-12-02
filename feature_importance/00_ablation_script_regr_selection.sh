#!/bin/bash

slurm_script="00_ablation_regression_selection_script.sh"

for data_name in "temperature" "performance" "parkinsons" "CCLE_PD_0325901"; do
    for split_seed in {1..3}; do
        for rf_seed in {1..3}; do
            sbatch $slurm_script $data_name $split_seed $rf_seed
            sleep 2 
        done
    done
done