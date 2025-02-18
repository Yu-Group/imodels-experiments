#!/bin/bash
#SBATCH --partition=jsteinhardt

# dataname="361260"
# seed=0
# clustertype="kmeans"
# clustermodel="linear"
# methodname="rf"

datafolder="fulldata"

source activate mdi
command="compile-results.py --dataname ${1} --seed ${2} --clustertype ${3} --clustermodel ${4} --methodname ${5} --datafolder $datafolder"
# command="compile-results.py --dataname $dataname --seed $seed --clustertype $clustertype --clustermodel $clustermodel --methodname $methodname --datafolder $datafolder"

# Execute the command
python $command