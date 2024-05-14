#!/bin/bash

slurm_script="feature_ranking.sh"

for rep in {1..10}
do
    sbatch $slurm_script $rep  # Submit SLURM job using the specified script
done