#!/bin/bash

slurm_script="correlation.sh"

pve=0.1

# try with more rhos to examine behavior
rhos=(0.5 0.6 0.7 0.8 0.85 0.9 0.95 0.99)

for rep in {1..50}
do
    for rho in ${rhos[@]}
    do
        sbatch $slurm_script $rep $pve $rho  # submit SLURM job using the specified script
    done
done