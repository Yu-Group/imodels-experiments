#!/bin/bash

slurm_script="01_auroc_regression_script_lss.sh" #"01_auroc_regression_script_linear_concept_shift.sh"

for rep in {1..10}
do
    sbatch $slurm_script $rep  # Submit SLURM job using the specified script
    sleep 2
done