#!/bin/bash
#SBATCH --job-name=godst
#SBATCH --mail-type=ALL
#SBATCH --mail-user=omer_ronen@berkeley.edu
#SBATCH -o godst.out #File to which standard out will be written
#SBATCH -e godst.err #File to which standard err will be written
python -m viz