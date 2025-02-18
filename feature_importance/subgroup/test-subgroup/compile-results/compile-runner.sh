#!/bin/bash
#SBATCH --partition=jsteinhardt

slurm_script="compile-results.sh"

# ids=("Dutch_drinking_inh" "Dutch_drinking_wm" "Dutch_drinking_sha"
#      "Brazil_health_heart" "Brazil_health_stroke" "Korea_grip"
#      "China_glucose_women2" "China_glucose_men2" "Spain_Hair" "China_HIV")
# clusttype=("hierarchical" "kmeans")
# modeltype=("tree" "linear")
modeltype=("linear")
# seeds=(1 2 3 4 5)
# ids=("361253" "361254" "361259" "361260" "361622" "361242")
ids=("361260")
clusttype=("kmeans")
# modeltype=("linear")
# seeds=(0 1 2 3 4 5 42 123 1234 2025)
seeds=(0 1 2 3 4)
# seeds=(0)
# methods=("gb" "rf")
methods=("rf")
# ids=(361234 361235 361236 361237 361241 361242 361243 361244
#      361247 361249 361250 361251 361252 361253 361254 361255
#      361256 361257 361258 361259 361260 361261 361264 361266
#      361267 361268 361269 361272 361616 361617 361618 361619
#      361621 361622 361623)

for method in "${methods[@]}"; do
    for seed in "${seeds[@]}"; do
        for id in "${ids[@]}"; do
            for clust in "${clusttype[@]}"; do
                for model in "${modeltype[@]}"; do
                    sbatch $slurm_script $id $seed $clust $model $method # Submit SLURM job using the specified script
                done
            done
        done
    done
done
