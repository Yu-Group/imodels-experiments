#!/bin/bash
#SBATCH --partition=jsteinhardt

slurm_script="correlation.sh"

for rep in {1..50}
do
    for pve in {0.1,0.4}
    do
        for rho in {0.5,0.6,0.7,0.8,0.9,0.99}
        do
            sbatch $slurm_script $rep $pve $rho  # Submit SLURM job using the specified script
        done
    done
done