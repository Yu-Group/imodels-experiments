#!/bin/bash
#SBATCH --partition=jsteinhardt

slurm_script="classification-job.sh"

classification_ids=("361063" "361069" "361062" "9978" "361071" "43")
n_ests=("50" "100" "500" "1000")
min_samples_leafs=("1" "5" "10")
max_features=("0.1" "sqrt" "1")

for dataid in "${classification_ids[@]}"; do
    for n_est in "${n_ests[@]}"; do
        for min_samples_leaf in "${min_samples_leafs[@]}"; do
            for max_feature in "${max_features[@]}"; do
                    sbatch $slurm_script $dataid $n_est $min_samples_leaf $max_feature # submit SLURM job using the specified script
            done
        done
    done
done