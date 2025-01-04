#!/bin/bash

slurm_script="00_ablation_regression_ranking_script.sh"

for data_name in "openml_361242" "openml_361251" "openml_361253" "openml_361254" "openml_361256" "openml_361259" "openml_361260" "openml_361622"; do # "openml_361242" "openml_361251" "openml_361253" "openml_361254" "openml_361256" "openml_361259" "openml_361260" "openml_361622";
    for dgp in "linear" "lss" "linear_lss" "linear_poly" "poly"; do #"linear" "lss" "linear_lss" "linear_poly" "poly"
        for feature_seed in {1..5}; do
            for error_seed in {1..2}; do
                sbatch $slurm_script $data_name $dgp $error_seed $feature_seed
                sleep 5
            done
        done
    done
done
