#!/bin/bash
#SBATCH --partition=yugroup
#SBATCH --cpus-per-task=4

dataname="361260"
clustertype="kmeans"
clustermodel="linear"

source activate mdi
command="compile-rf-results.py --dataname $dataname --seed ${1} --clustertype $clustertype --clustermodel $clustermodel" # --methodname ${5} --datafolder $datafolder"

# execute the command
python $command