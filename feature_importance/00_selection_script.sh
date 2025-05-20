#!/bin/bash
#SBATCH --partition=low
source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="00_run_selection.py --nreps 1 --config mdi_local.real_data_${1}_${2} --split_seed ${3} --sample_seed ${4} --ignore_cache --create_rmd --folder_name ${2}_selection2"

# Execute the command
python $command