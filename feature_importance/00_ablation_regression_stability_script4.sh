#!/bin/bash
#SBATCH --mail-user=zhongyuan_liang@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=yugroup
source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="00_run_ablation_regression_stability.py --nreps 1 --config mdi_local.real_data_regression_CCLE_PD_0325901_retrain --split_seed ${1} --ignore_cache --create_rmd --folder_name CCLE_PD_0325901_stability"

# Execute the command
python $command