#!/bin/bash
#SBATCH --partition=yugroup
source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="00_run_ablation_regression_selection.py --nreps 1 --config mdi_local.real_data_regression_${1} --split_seed ${2} --rf_seed ${3} --ignore_cache --create_rmd --folder_name ${1}_selection --fit_model True"

# Execute the command
python $command