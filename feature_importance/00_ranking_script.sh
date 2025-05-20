#!/bin/bash
#SBATCH --partition=high
source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="00_run_ranking.py --nreps 1 --config mdi_local.real_data_${1}_${2}_${3} --split_seed 42 --error_seed 0 --feature_seed ${4} --sample_seed ${5} --ignore_cache --create_rmd --folder_name ${2}_${3}2"

# Execute the command
python $command