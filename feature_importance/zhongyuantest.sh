#!/bin/bash
#SBATCH --partition=jsteinhardt

source activate mdi
command="zhongyuantest.py"

# Execute the command
python $command