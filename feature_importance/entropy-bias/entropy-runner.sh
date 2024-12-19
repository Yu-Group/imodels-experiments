#!/bin/bash

slurm_script="entropy.sh"
types=("classification" "regression")


for rep in {1..50}
do
    for value in {50,100,250,500,1000}
    do
        for task in "${types[@]}"
        do
            sbatch $slurm_script $rep $task $value  # Submit SLURM job using the specified script
        done
    done
done