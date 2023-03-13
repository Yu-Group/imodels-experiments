#!/bin/bash
#SBATCH --account=co_stat
#SBATCH --partition=savio
#SBATCH --time=12:00:00
#
#SBATCH --nodes=1

module load python/3.7
module load r

source activate r2f

python ../03_run_prediction_simulations.py --nreps 50 --config ${1} --split_seed 12345 --parallel --mode ${2} --nosave_cols "prediction_model"
