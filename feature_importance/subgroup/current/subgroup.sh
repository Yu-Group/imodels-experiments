#!/bin/bash
#SBATCH --cpus-per-task=32

seed=1
# dataid=361247
clustertype="kmeans"
njobs=32

source activate mdi
command="subgroup-incase.py --seed $seed --dataid ${1} --clustertype $clustertype --njobs $njobs"
# command="subgroup-incase.py --seed $seed --dataid $dataid --clustertype $clustertype --njobs $njobs"

# Execute the command
python $command