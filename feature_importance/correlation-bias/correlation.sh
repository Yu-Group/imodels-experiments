#!/bin/bash
#SBATCH --partition=yugroup
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zachrewolinski@berkeley.edu
#SBATCH --output=slurm_output/correlation-%j.out

njobs=8

source activate mdi
command="correlation_pipeline.py --seed ${1} --pve ${2} --rho ${3} --njobs $njobs"

# execute the command
python $command