#!/bin/bash

slurm_script="01_ablation_conditional_classification_script.sh"

for rep in {1..10}
do
    sbatch $slurm_script $rep  # Submit SLURM job using the specified script
    sleep 2
done