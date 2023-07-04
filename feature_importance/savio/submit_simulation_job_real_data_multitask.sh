#!/bin/bash
#SBATCH --account=co_stat
#SBATCH --partition=savio
#SBATCH --time=48:00:00
#
#SBATCH --nodes=1

module load python/3.7
module load r

source activate r2f

python ../02_run_importance_real_data.py --nreps 32 --config ${1} --response_idx ${2} --split_seed 12345 --parallel --nosave_cols "prediction_model"
