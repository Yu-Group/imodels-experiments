#!/bin/bash

slurm_script="regression-job.sh"

# choose SARCOS, Puma Robot, Wave Energy, and Super Conductivity due to varied
# feature set sizes
regression_ids=("361254" "361259" "361253" "361242")
n_ests=("50" "100" "500" "1000")
min_samples_leafs=("1" "5" "10")
max_features=("0.1" "0.33" "1")

for dataid in "${regression_ids[@]}"; do
    for n_est in "${n_ests[@]}"; do
        for min_samples_leaf in "${min_samples_leafs[@]}"; do
            for max_feature in "${max_features[@]}"; do
                    sbatch $slurm_script $dataid $n_est $min_samples_leaf $max_feature # submit SLURM job using the specified script
            done
        done
    done
done