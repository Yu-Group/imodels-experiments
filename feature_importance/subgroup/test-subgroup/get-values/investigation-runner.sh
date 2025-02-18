#!/bin/bash
#SBATCH --partition=yugroup

slurm_script="investigation.sh"

# ids=("361242" "361253" "361254" "361259" "361260" "361622")
# ids=("361622")
ids=(361260)
# ids=("Dutch_drinking_inh" "Dutch_drinking_wm" "Dutch_drinking_sha"
#      "Brazil_health_heart" "Brazil_health_stroke" "Korea_grip"
#      "China_glucose_women2" "China_glucose_men2" "Spain_Hair" "China_HIV")
# ids=(361234 361235 361236 361237 361241 361242 361243 361244
#      361247 361249 361250 361251 361252 361253 361254 361255
#      361256 361257 361258 361259 361260 361261 361264 361266
#      361267 361268 361269 361272 361616 361617 361618 361619
#      361621 361622 361623)
# seeds=(0 1 2 3 4)
methods=("gb")
seeds=(5 42 2025 123 1234)
# seeds=(0)

for method in "${methods[@]}"; do
    for id in "${ids[@]}"; do
        for seed in "${seeds[@]}"; do
            sbatch $slurm_script $id $seed $method # Submit SLURM job using the specified script
        done
    done
done


