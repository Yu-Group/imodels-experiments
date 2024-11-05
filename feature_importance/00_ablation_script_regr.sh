#!/bin/bash

slurm_script="00_ablation_regression_script.sh"

for rep in {1..2}
do
    sbatch $slurm_script $rep  # Submit SLURM job using the specified script
    sleep 2
done