#!/bin/bash
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

classification=0

source activate mdi
# command="runtime_analysis.py --dataid ${1} --classification ${classification} --n_estimators ${2} --min_samples_leaf ${3} --max_features ${4}"
command="runtime_analysis.py --dataid 361260 --classification ${classification} --n_estimators 100 --min_samples_leaf 10 --max_features 0.1"
python $command