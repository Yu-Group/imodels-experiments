#!/bin/bash
#SBATCH --account=co_stat
#SBATCH --partition=savio
#SBATCH --time=12:00:00
#
#SBATCH --nodes=1
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tiffany.tang@berkeley.edu

module load python/3.7
module load r

source activate r2f

python 01_run_simulations.py --nreps 20 --config ${1} --split_seed 12345  --show_vars 100 --parallel --create_rmd --ignore_cache
