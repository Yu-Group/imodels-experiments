#!/bin/bash

dataname="361260"
clustertype="kmeans"
clustermodel="linear"

source activate mdi
command="compile-results.py --dataname $dataname --seed ${1} --clustertype $clustertype --clustermodel $clustermodel" # --methodname ${5} --datafolder $datafolder"

# execute the command
python $command