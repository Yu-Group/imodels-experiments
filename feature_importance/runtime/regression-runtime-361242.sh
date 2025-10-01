#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

id="361242"
classification=0

source activate mdi
command="runtime_analysis.py --dataid ${id} --classification ${classification} --n_estimators ${1} --min_samples_leaf ${2} --max_features ${3}"
python $command
