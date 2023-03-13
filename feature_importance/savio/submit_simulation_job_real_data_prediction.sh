#!/bin/bash
#SBATCH --account=co_stat
#SBATCH --partition=savio
#SBATCH --time=24:00:00
#
#SBATCH --nodes=1

module load python/3.7
module load r

source activate r2f

python ../04_run_prediction_real_data.py --nreps 32 --config ${1} --split_seed 12345 --parallel --mode ${2} --subsample_n 10000 --nosave_cols "prediction_model"
