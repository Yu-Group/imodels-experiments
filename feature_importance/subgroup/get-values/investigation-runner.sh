#!/bin/bash
#SBATCH --partition=yugroup

slurm_script="investigation.sh"

id=361260
seeds=(0 1 2 3 4)
method="rf"

for seed in "${seeds[@]}"; do
    sbatch $slurm_script $id $seed $method # submit SLURM job using the specified script
done
