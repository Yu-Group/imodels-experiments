#!/bin/bash
#SBATCH --partition=jsteinhardt

dataname="361260"
clustertype="kmeans"
clustermodel="linear"
# methodname="rf"
# datafolder="fulldata"

source activate mdi
command="compile-results.py --dataname $dataname --seed ${1} --clustertype $clustertype --clustermodel $clustermodel" # --methodname ${5} --datafolder $datafolder"
# command="compile-results.py --dataname $dataname --seed $seed --clustertype $clustertype --clustermodel $clustermodel --methodname $methodname --datafolder $datafolder"

# Execute the command
python $command