#!/bin/bash
#SBATCH --partition=jsteinhardt
source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="00_run_classification_selection_vary_sample_size_no_aggregation_no_retrain.py --nreps 1 --config mdi_local.real_data_classification_${1} --split_seed ${2} --sample_seed ${3} --rf_seed 1 --ignore_cache --create_rmd --folder_name ${1}_selection_vary_sample_size_no_aggregation_no_retrain"

# Execute the command
python $command