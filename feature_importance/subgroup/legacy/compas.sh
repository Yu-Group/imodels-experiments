#!/bin/bash
#SBATCH --mail-user=zachrewolinski@berkeley.edu
#SBATCH --mail-type=ALL

source activate mdi
command="compas.py --nreps 1"

# Execute the command
python $command