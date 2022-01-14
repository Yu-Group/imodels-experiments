#!/bin/bash

#SBATCH --job-name=bart
#SBATCH --mail-type=ALL
#SBATCH --mail-user=omer_ronen@berkeley.edu
#SBATCH -o bart.out #File to which standard out will be written
#SBATCH -e bart.err #File to which standard err will be written
python -m viz