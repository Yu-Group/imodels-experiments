#!/bin/bash

slurm_script="00_ablation_regression_ranking_script.sh"

for data_name in "temperature" "performance" "parkinsons" "CCLE_PD_0325901"; do
    for dgp in "linear" "lss" "poly"; do
        for y_seed in {1..10}; do
            sbatch $slurm_script $data_name $dgp $y_seed
            sleep 2
        done
    done
done
