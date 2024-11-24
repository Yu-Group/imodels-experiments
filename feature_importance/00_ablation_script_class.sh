#!/bin/bash

slurm_script="00_ablation_classification_script4.sh"

for split_seed in {1..3}; do
    for rf_seed in {1..5}; do
        sbatch $slurm_script $split_seed $rf_seed  # Submit SLURM job with both split_seed and rf_seed as arguments
        sleep 2
    done
done
