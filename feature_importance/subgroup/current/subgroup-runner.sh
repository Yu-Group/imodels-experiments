#!/bin/bash
#SBATCH --partition=yss

slurm_script="subgroup.sh"

ids=(361242 361251 361253 361260 361259 361256 361254 361622)

for id in "${ids[@]}"; do
    sbatch $slurm_script $id  # Submit SLURM job using the specified script
done
