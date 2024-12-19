#!/bin/bash
#SBATCH --cpus-per-task=32

datasource="function"
dataname="ccle"
# seed=1
njobs=32
use_test=1

source activate mdi
# command="agglomerative_subgroups.py --seed $seed --datasource $datasource --dataname $dataname --use_test $use_test --njobs $njobs"
command="agglomerative_subgroups.py --seed ${1} --datasource $datasource --dataname $dataname --use_test $use_test --njobs $njobs"

# Execute the command
python $command