#!/bin/bash
#SBATCH --partition=high

source activate mdi
# command="zhongyuantest.py -w shadowfax"
command="zhongyuantest.py"

# Execute the command
python $command