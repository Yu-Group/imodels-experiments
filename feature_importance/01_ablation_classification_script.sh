#!/bin/bash
#SBATCH --mail-user=zhongyuan_liang@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=yugroup

source activate mdi
command="01_run_ablation_classification.py --nreps 1 --config mdi_local.real_data_classification --split_seed ${1} --ignore_cache --create_rmd --result_name Diabetes_classification_parallel"

# Execute the command
python $command