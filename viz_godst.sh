#!/bin/bash
JOB_NAME=$0
#SBATCH --job-name=$JOB_NAME
#SBATCH --mail-type=ALL
#SBATCH --mail-user=omer_ronen@berkeley.edu
#SBATCH -o $JOB_NAME.out #File to which standard out will be written
#SBATCH -e $JOB_NAME.err #File to which standard err will be written
python -m viz