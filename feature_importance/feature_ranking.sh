#!/bin/bash
#SBATCH --mail-user=zachrewolinski@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=yugroup

source activate mdi
command="run_importance_local_sims.py --nreps 1 --config mdi_local.real_x_sim_y --split_seed 1 --ignore_cache --create_rmd --result_name feature_ranking"

# Execute the command
python $command