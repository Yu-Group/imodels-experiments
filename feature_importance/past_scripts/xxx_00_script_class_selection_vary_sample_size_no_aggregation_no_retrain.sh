#!/bin/bash

slurm_script="00_classification_selection_vary_sample_size_script_no_aggregation_no_retrain.sh"

for data_name in "openml_43" "openml_9946" "openml_9978" "openml_146819" "openml_361062" "openml_361070"; do #"openml_43" "openml_9946" "openml_9978" "openml_146819" "openml_361062" "openml_361070"
    for split_seed in {1..2}; do
        for sample_seed in {1..5}; do
            sbatch $slurm_script $data_name $split_seed $sample_seed
            sleep 1
        done
    done
done
