#!/bin/bash

slurm_script="01_auroc_regression_script_linear.sh"

for rep in {1..3}
do
    sbatch $slurm_script $rep  # Submit SLURM job using the specified script
    sleep 2
done