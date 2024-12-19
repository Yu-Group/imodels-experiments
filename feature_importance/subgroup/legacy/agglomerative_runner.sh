#!/bin/bash

slurm_script="agglomerative_subgroups.sh"

for rep in {1..20}
do
    sbatch $slurm_script $rep  # Submit SLURM job using the specified script
done