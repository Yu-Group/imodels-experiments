#!/bin/bash
#SBATCH --partition=low
source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="00_run_regression_ranking_vary_sample_size.py --nreps 1 --config mdi_local.real_data_regression_${1}_${2} --split_seed 1 --error_seed 1 --feature_seed ${3} --sample_seed ${4} --ignore_cache --create_rmd --folder_name ${1}_${2}_vary_sample_size_2000"

# Execute the command
python $command