#!/bin/bash
#SBATCH --mail-user=zhongyuan_liang@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=yugroup

source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="01_run_feature_ranking_simulation.py --nreps 1 --config mdi_local.synthetic_data_lss --x_seed ${1} --y_seed ${2} --split_seed ${3} --ignore_cache --create_rmd --folder_name lss --fit_model True --dgp_fi lss"

# Execute the command
python $command