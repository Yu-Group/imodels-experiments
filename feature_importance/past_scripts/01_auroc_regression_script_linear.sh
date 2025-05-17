#!/bin/bash
#SBATCH --mail-user=zhongyuan_liang@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=yugroup

source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="01_run_feature_ranking_simulation_linear.py --nreps 1 --config mdi_local.synthetic_data_linear --x_seed ${1} --y_seed ${2} --split_seed ${3} --ignore_cache --create_rmd --folder_name linear --fit_model True --dgp_fi linear"

# Execute the command
python $command