#!/bin/bash
#SBATCH --mail-user=zhongyuan_liang@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=yugroup

source activate mdi
# Need to specify --result_name --ablate_features(default all features) --fitted(default not fitted)
command="01_run_auroc_synthetic.py --nreps 1 --config mdi_local.synthetic_data_lss --split_seed ${1} --ignore_cache --create_rmd --folder_name lss_synthetic --fit_model True"

# Execute the command
python $command