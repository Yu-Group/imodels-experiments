#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --cpus-per-task=8

njobs=8

source activate mdi
command="correlation_pipeline.py --seed ${1} --pve ${2} --rho ${3} --njobs $njobs"

# Execute the command
python $command