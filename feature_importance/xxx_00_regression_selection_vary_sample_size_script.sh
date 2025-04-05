#!/bin/bash
#SBATCH --partition=high
source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="00_run_regression_selection_vary_sample_size.py --nreps 1 --config mdi_local.real_data_regression_${1} --split_seed ${2} --sample_seed ${3} --rf_seed 1 --ignore_cache --create_rmd --folder_name ${1}_selection_vary_sample_size"

# Execute the command
python $command