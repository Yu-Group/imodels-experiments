#!/bin/bash
#SBATCH --partition=jsteinhardt
source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="00_run_classification_selection.py --nreps 1 --config mdi_local.real_data_classification_${1} --split_seed ${2} --rf_seed ${3} --ignore_cache --create_rmd --folder_name ${1}_selection"

# Execute the command
python $command