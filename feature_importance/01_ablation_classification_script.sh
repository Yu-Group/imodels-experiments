#!/bin/bash
#SBATCH --mail-user=zhongyuan_liang@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=yugroup

source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="01_run_ablation_classification.py --nreps 1 --config mdi_local.real_data_classification --split_seed ${1} --ignore_cache --create_rmd --ablate_features 20 --result_name Juvenile --fitted True"
# Execute the command
python $command