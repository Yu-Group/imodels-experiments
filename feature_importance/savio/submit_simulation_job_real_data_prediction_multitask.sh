#!/bin/bash
#SBATCH --account=co_stat
#SBATCH --partition=savio
#SBATCH --time=24:00:00
#
#SBATCH --nodes=1

module load python/3.7
module load r

source activate r2f

python ../04_run_prediction_real_data.py --nreps 32 --config ${1} --split_seed 12345 --parallel --response_idx ${2} --mode ${3} --nosave_cols "prediction_model"
