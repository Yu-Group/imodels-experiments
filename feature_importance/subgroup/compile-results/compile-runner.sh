#!/bin/bash
#SBATCH --partition=jsteinhardt

slurm_script="compile-results.sh"
modeltype=("linear")
ids=("361260")
clusttype=("kmeans")
seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
    sbatch $slurm_script $seed # Submit SLURM job using the specified script
done
