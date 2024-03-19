#!/bin/bash
#SBATCH --mail-user=zhongyuan_liang@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=yugroup

command="01_run_ablation_regression.py --nreps 1 --config mdi_local.real_data_regression --split_seed ${1} --ignore_cache --create_rmd --result_name satellite_image_parallel"

# Execute the command
python $command