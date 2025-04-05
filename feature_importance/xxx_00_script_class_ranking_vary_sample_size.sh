#!/bin/bash

slurm_script="00_classification_ranking_vary_sample_size_script.sh"

for data_name in "openml_43" "openml_9946" "openml_9978" "openml_146819" "openml_361062" "openml_361070"; do #"openml_43" "openml_9946" "openml_9978" "openml_146819" "openml_361062" "openml_361070"
    for dgp in "logistic_linear" "logistic_linear_lss" "logistic_interaction"; do #"logistic_linear" "logistic_linear_lss" "logistic_interaction"
        for feature_seed in {1..5}; do
            for sample_seed in {1..6}; do
                sbatch $slurm_script $data_name $dgp $feature_seed $sample_seed
                sleep 1
            done
        done
    done
done