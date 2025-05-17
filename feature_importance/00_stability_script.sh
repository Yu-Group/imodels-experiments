#!/bin/bash
#SBATCH --partition=epurdom #low
source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="00_run_stability.py --nreps 1 --config mdi_local.real_data_${1}_${2}_stability --split_seed ${3} --sample_seed ${4} --ignore_cache --create_rmd --folder_name ${2}_stability"

# Execute the command
python $command