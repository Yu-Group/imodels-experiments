#!/bin/bash

slurm_script="01_selection_script.sh"

# for data_name in "openml_361242" "openml_361243" "openml_361253" "openml_361254" "openml_361259" "openml_361260"; do
#     for split_seed in {1..4}; do
#         for sample_seed in {1..5}; do
#             sbatch $slurm_script "regression" $data_name $split_seed $sample_seed
#         done
#     done
# done

for data_name in "openml_43" "openml_9978" "openml_361062" "openml_361063" "openml_361069" "openml_361071"; do
    for split_seed in {1..4}; do
        for sample_seed in {1..5}; do
            sbatch $slurm_script "classification" $data_name $split_seed $sample_seed
        done
    done
done
