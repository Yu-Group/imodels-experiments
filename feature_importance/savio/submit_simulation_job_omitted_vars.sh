#!/bin/bash
#SBATCH --account=co_stat
#SBATCH --partition=savio
#SBATCH --time=12:00:00
#
#SBATCH --nodes=1

module load python/3.7
module load r

source activate r2f

python ../01_run_simulations.py --nreps 50 --config ${1} --split_seed 12345 --omit_vars 0,1 --parallel
