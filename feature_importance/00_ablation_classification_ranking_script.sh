#!/bin/bash
#SBATCH --partition=yugroup
source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="00_run_feature_ranking_simulation_classification.py --nreps 1 --config mdi_local.real_data_classification_${1}_${2} --split_seed 1 --error_seed ${3} --feature_seed ${4} --ignore_cache --create_rmd --folder_name ${1}_${2}"

# Execute the command
python $command