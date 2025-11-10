#!/bin/bash
#SBATCH --partition=yugroup

njobs=8
# seed=1
# pve=0.1
# rho=0.5

source activate mdi
command="correlation_pipeline.py --seed ${1} --pve ${2} --rho ${3} --njobs $njobs"
# command="correlation_pipeline_new.py --seed ${1} --pve ${2} --rho ${3} --njobs $njobs"
# command="correlation_pipeline_gb.py --seed $seed --pve $pve --rho $rho --njobs $njobs"

# Execute the command
python $command