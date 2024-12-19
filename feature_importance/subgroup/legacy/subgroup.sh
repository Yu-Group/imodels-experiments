#!/bin/bash
#SBATCH --mail-user=zachrewolinski@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=16

datasource="imodels"
dataname="compas_two_year_clean"

source activate mdi
command="evaluate_subgroups.py --seed ${1} --datasource $datasource --dataname $dataname"

# Execute the command
python $command