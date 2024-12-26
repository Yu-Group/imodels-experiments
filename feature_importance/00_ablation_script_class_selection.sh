#!/bin/bash

slurm_script="00_ablation_classification_selection_script.sh"

for data_name in "openml_9946"; do #"openml_43" "openml_3917" "openml_9946" "openml_9978" "openml_146819" "openml_167120"
    for split_seed in {1..3}; do
        for rf_seed in {1..3}; do
            sbatch $slurm_script $data_name $split_seed $rf_seed
            sleep 5
        done
    done
done