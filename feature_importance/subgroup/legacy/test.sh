#!/bin/bash
#SBATCH --mail-user=zachrewolinski@berkeley.edu
#SBATCH --mail-type=ALL

source activate mdi
command="test.py --nreps 1"

# Execute the command
python $command