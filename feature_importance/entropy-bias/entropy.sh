#!/bin/bash
#SBATCH --cpus-per-task=8

njobs=8
test=1

source activate mdi
command="entropy_pipeline.py --seed ${1} --task ${2} --n ${3} --test $test --njobs $njobs"

# Execute the command
python $command