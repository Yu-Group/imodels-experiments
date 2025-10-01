#!/bin/bash
#SBATCH --partition=jsteinhardt

slurm_script="regression-runtime-361259.sh"

n_ests=("25" "50" "100" "200" "500" "1000")
min_samples_leafs=("1" "3" "5" "10")
max_features=("0.1" "0.33" "0.5" "1")

for n_est in "${n_ests[@]}"; do
    for min_samples_leaf in "${min_samples_leafs[@]}"; do
        for max_feature in "${max_features[@]}"; do
                sbatch $slurm_script $n_est $min_samples_leaf $max_feature # submit SLURM job using the specified script
        done
    done
done