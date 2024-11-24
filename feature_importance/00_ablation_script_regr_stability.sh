#!/bin/bash

slurm_script="00_ablation_regression_stability_script4.sh"

for split_seed in {1..3}; do
    sbatch $slurm_script $split_seed # Submit SLURM job with both split_seed and rf_seed as arguments
    sleep 2
done