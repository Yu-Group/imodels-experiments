#!/bin/bash

slurm_script="subgroup.sh"

for rep in {1..5}
do
    sbatch $slurm_script $rep  # Submit SLURM job using the specified script
done