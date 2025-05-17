#!/bin/bash
#SBATCH --partition=high
source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="00_run_regression_ranking.py --nreps 1 --config mdi_local.real_data_regression_${1}_${2} --split_seed 42 --error_seed 0 --feature_seed ${3} --sample_seed ${4} --ignore_cache --create_rmd --folder_name ${1}_${2}"

# Execute the command
python $command