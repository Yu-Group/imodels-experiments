#!/bin/bash
#SBATCH --mail-user=zhongyuan_liang@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=yugroup
source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="00_run_ablation_regression_retrain.py --nreps 1 --config mdi_local.real_data_regression_performance_retrain --split_seed ${1} --rf_seed ${2} --ignore_cache --create_rmd --folder_name performance_retrain --fit_model True --absolute_masking True"

# Execute the command
python $command