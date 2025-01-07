#!/bin/bash
#SBATCH --partition=yss
#SBATCH --cpus-per-task=8

# seed=1
# dataid=361268
pipeline=2
clustertype="hierarchical"
standardize=1

source activate mdi
command="subgroup.py --seed ${2} --dataid ${1} --pipeline $pipeline --clustertype $clustertype --standardize $standardize"
# command="subgroup.py --seed $seed --dataid $dataid --pipeline $pipeline --clustertype $clustertype"

# Execute the command
python $command