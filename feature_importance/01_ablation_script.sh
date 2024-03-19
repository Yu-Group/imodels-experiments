#!/bin/bash

slurm_script="01_ablation_regression_script.sh"

rep=1
sbatch $slurm_script $rep

# for rep in {1..5}
# do
#     sbatch $slurm_script $rep  # Submit SLURM job using the specified script
# done