#!/bin/bash

slurm_script="00_classification_selection_script.sh"

for data_name in "openml_361062"; do #"openml_43" "openml_9946" "openml_9978" "openml_146819" "openml_361062" "openml_361070"
    for split_seed in {1..5}; do
        for rf_seed in {1..3}; do
            sbatch $slurm_script $data_name $split_seed $rf_seed
            sleep 1
        done
    done
done
