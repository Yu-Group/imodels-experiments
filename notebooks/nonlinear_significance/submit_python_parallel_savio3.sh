#!/bin/bash
#SBATCH --account=co_stat
#SBATCH --partition=savio3
#SBATCH --time=6:00:00
#
#SBATCH --nodes=1

module load python/3.7
module load r

source activate r2f

python 01_run_simulations.py --nreps 32 --config ${1} --split_seed 331  --r2 --show_vars 100 --parallel --create_rmd
