#!/bin/bash
#SBATCH --mail-user=zachrewolinski@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=32
#SBATCH --exclusive

source activate mdi
command="gene_importance.py"

# Execute the command
python $command