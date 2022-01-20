#!/bin/bash

#SBATCH --job-name=gosdt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=omer_ronen@berkeley.edu

/accounts/campus/omer_ronen/.conda/envs/tree_shrink/bin/python -m 01_fit_models --config shrinkage --classification --split_seed $1 --reg $2