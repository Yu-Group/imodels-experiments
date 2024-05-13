#!/bin/bash

slurm_script="01_ablation_classification_script.sh"

for rep in {1..5}
do
    sbatch $slurm_script $rep  # Submit SLURM job using the specified script
done