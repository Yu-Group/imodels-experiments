#!/bin/bash

slurm_script="00_ablation_classification_ranking_script.sh"

for data_name in "openml_43" "openml_3917" "openml_9946" "openml_9978" "openml_146819" "openml_167120" ; do #"openml_43" "openml_3917" "openml_9946" "openml_9978" "openml_146819" "openml_167120" 
    for dgp in "logistic_linear" "logistic_linear_lss" "logistic_poly"; do #"logistic_linear" "logistic_linear_lss" "logistic_poly"
        for feature_seed in {1..5}; do
            for error_seed in {1..2}; do
                sbatch $slurm_script $data_name $dgp $error_seed $feature_seed
                sleep 5
            done
        done
    done
done
