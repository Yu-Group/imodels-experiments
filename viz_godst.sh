#!/bin/bash

#SBATCH --job-name=gosdt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=omer_ronen@berkeley.edu
#SBATCH -o gosdt.out #File to which standard out will be written
#SBATCH -e gosdt.err #File to which standard err will be written
python -m 01_fit_models --config shrinkage --classification --split_seed $1