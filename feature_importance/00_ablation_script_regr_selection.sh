#!/bin/bash

slurm_script="00_ablation_regression_selection_script.sh"

for data_name in "CCLE_AZD_0530" "CCLE_AZD_0530_high_variance_x" "CCLE_PD_0325901" "CCLE_PD_0325901_high_variance_x" "openml_361242" "openml_361251" "openml_361253" "openml_361254" "openml_361256" "openml_361259" "openml_361260" "openml_361622"; do #"CCLE_AZD_0530" "CCLE_AZD_0530_high_variance_x" "CCLE_PD_0325901" "CCLE_PD_0325901_high_variance_x" "openml_361242" "openml_361251" "openml_361253" "openml_361254" "openml_361256" "openml_361259" "openml_361260" "openml_361622"
    for split_seed in {1..3}; do
        for rf_seed in {1..3}; do
            sbatch $slurm_script $data_name $split_seed $rf_seed
            sleep 5
        done
    done
done