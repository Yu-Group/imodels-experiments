#!/bin/bash
#SBATCH --partition=yugroup

# dataname="361260"
# seed=4

source activate mdi
command="investigation.py --dataname ${1} --seed ${2} --method ${3}"
# command="investigation.py --dataname ${dataname} --seed ${seed}"

# Execute the command
python $command