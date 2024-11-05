#!/bin/bash

slurm_script="01_ablation_regression_script_average3.sh"

for rep in {1..2}
do
    sbatch $slurm_script $rep  # Submit SLURM job using the specified script
    sleep 2
done