#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

classification=0

source activate mdi
command="runtime_analysis.py --dataid ${1} --classification ${classification} --n_estimators ${2} --min_samples_leaf ${3} --max_features ${4}"
python $command