#!/bin/bash
#SBATCH --partition=yugroup

slurm_script="knn.sh"

ids=("43" "361062" "361071" "9978" "361069" "361063")
# kvals=("1" "3" "5" "7" "9")
kvals=("1")
# nbr_dists=("l1" "l2")
# cfact_dists=("l1" "l2")
nbr_dists=("l1")
cfact_dists=("l1")
use_preds=("0" "1")

for id in "${ids[@]}"; do
    for k in "${kvals[@]}"; do
        for nbr_dist in "${nbr_dists[@]}"; do
            for cfact_dist in "${cfact_dists[@]}"; do
                for pred in "${use_preds[@]}"; do
                # echo "Running with id: $id, k: $k, nbr_dist: $nbr_dist, cfact_dist: $cfact_dist"
                    sbatch $slurm_script $id $k $nbr_dist $cfact_dist $pred # Submit SLURM job using the specified script
                done
            done
        done
    done
done


