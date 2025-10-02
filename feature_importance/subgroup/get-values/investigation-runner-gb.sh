#!/bin/bash

slurm_script="investigation-gb.sh"

id=361260
seeds=(0 1 2 3 4)
method="gb"

for seed in "${seeds[@]}"; do
    sbatch $slurm_script $id $seed $method # submit SLURM job using the specified script
done
