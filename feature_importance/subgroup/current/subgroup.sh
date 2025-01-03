#!/bin/bash
#SBATCH --partition=yss
#SBATCH --cpus-per-task=8

seed=1
# dataid=361247
pipeline=1
clustertype="kmeans"
njobs=32

source activate mdi
command="subgroup.py --seed $seed --dataid ${1} --pipeline $pipeline --clustertype $clustertype --njobs $njobs"
# command="subgroup-incase.py --seed $seed --dataid $dataid --clustertype $clustertype --njobs $njobs"

# Execute the command
python $command