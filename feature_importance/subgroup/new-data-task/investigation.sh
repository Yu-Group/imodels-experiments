#!/bin/bash
#SBATCH --partition=yugroup

# dataname="361260"
# seed=4
gender=0
dataname="abalone"

source activate mdi
command="investigation.py --dataname ${dataname} --seed ${1} --method ${2} --gender ${gender}"
# command="investigation.py --dataname ${dataname} --seed ${seed}"

# Execute the command
python $command