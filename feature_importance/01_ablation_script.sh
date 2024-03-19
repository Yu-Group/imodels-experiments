#!/bin/bash
#SBATCH --mail-user=zhongyuan_liang@berkeley.edu
#SBATCH --mail-type=ALL

# Define the SLURM submission script name
slurm_script="01_ablation_regression_script.sh"  # Replace {slurm_submission_script} with your actual script name

# Loop to submit SLURM job 10 times
for rep in {1..10}
do
    sbatch $slurm_script  # Submit SLURM job using the specified script
done