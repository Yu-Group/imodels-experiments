#!/bin/bash
#SBATCH --partition=yugroup
#SBATCH --cpus-per-task=16

source activate mdi
command="pecarn.py"

# Execute the command
python $command