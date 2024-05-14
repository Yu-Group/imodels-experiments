#!/bin/bash
#SBATCH --mail-user=zachrewolinski@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=yugroup

source activate mdi
command="01_run_ablation_regression.py --nreps 1 --config mdi_local.real_data_regression --split_seed ${1} --ignore_cache --create_rmd --result_name satellite_image"

# Execute the command
python $command