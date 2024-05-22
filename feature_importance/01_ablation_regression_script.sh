#!/bin/bash
#SBATCH --mail-user=zhongyuan_liang@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=yugroup

source activate mdi
command="01_run_ablation_regression.py --nreps 1 --config mdi_local.real_data_regression --split_seed ${1} --ignore_cache --create_rmd --result_name diabetes_test_new"
#command="01_run_ablation_regression.py --nreps 1 --config mdi_local.real_data_regression --split_seed ${1} --ignore_cache --create_rmd --result_name CCLE_AZD0530_new --ablate_features 20"

# Execute the command
python $command