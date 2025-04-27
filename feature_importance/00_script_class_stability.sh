#!/bin/bash

slurm_script="00_classification_stability_script.sh"

for data_name in "openml_43" "openml_9978" "openml_361062" "openml_361063" "openml_361069" "openml_361071"; do
    for split_seed in {1..3}; do
        for sample_seed in {1..5}; do
            sbatch $slurm_script $data_name $split_seed $sample_seed
        done
    done
done

#"openml_43" "openml_9978" "openml_361062" "openml_361063" "openml_361069" "openml_361071"
