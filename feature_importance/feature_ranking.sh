#!/bin/bash
#SBATCH --mail-user=zachrewolinski@berkeley.edu
#SBATCH --mail-type=ALL

source activate mdi
command="ranking_importance_local_sims.py --nreps 1 --config mdi_local.real_x_sim_y.diabetes-regression.linear-model --split_seed 1 --ignore_cache --create_rmd --result_name diabetes-reg-linear"

# Execute the command
python $command