#!/bin/bash
#SBATCH --partition=yugroup
#SBATCH --cpus-per-task=4

source activate mdi
command="investigation.py --dataname ${1} --seed ${2} --method ${3}"

# Execute the command
python $command