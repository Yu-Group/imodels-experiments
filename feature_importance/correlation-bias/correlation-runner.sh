#!/bin/bash

slurm_script="correlation.sh"

pve=0.1

for rep in {1..50}
do
    for rho in {0.5,0.6,0.7,0.8,0.9,0.99}
    do
        sbatch $slurm_script $rep $pve $rho  # submit SLURM job using the specified script
    done
done