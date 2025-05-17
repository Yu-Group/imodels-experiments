#!/bin/bash

slurm_script="01_auroc_regression_script_linear.sh" #"01_auroc_regression_script_linear_concept_shift.sh"

for x_seed in {1..2}; do
    for y_seed in {1..5}; do
        for split_seed in {1..2}; do
            sbatch $slurm_script $x_seed $y_seed $split_seed  # Submit SLURM job using the specified script
            sleep 2
        done
    done
done