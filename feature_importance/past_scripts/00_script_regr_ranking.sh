#!/bin/bash

slurm_script="00_regression_ranking_script.sh"

for data_name in "openml_361242" "openml_361243" "openml_361253" "openml_361254" "openml_361259" "openml_361260"; do
    for dgp in "linear" "linear_lss" "interaction"; do
        for feature_seed in {1..10}; do
            for sample_seed in {1..3}; do
                sbatch $slurm_script $data_name $dgp $feature_seed $sample_seed
            done
        done
    done
done

