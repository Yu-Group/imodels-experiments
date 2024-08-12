#!/bin/bash
#SBATCH --mail-user=zhongyuan_liang@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=yugroup

source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="01_run_ablation_regression.py --nreps 1 --config mdi_local.synthetic_data_linear --split_seed 0 --simulation_seed ${1} --ignore_cache --create_rmd --folder_name linear_synthetic --fit_model True --positive_masking True --absolute_masking True --negative_masking True"

# Execute the command
python $command