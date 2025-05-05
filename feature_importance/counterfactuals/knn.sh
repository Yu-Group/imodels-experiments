#!/bin/bash
#SBATCH --partition=yugroup

src="openml"
id=361062
kval=3
nbr_dist="l2"
cfact_dist="l2"
use_preds=1

source activate mdi
command="knn.py --datasource ${src} --dataid ${1} --k ${2} --nbr_dist ${3} --cfact_dist ${4} --use_preds ${5}"
# command="knn.py --datasource ${src} --dataid ${id} --k ${kval}  --nbr_dist ${nbr_dist} --cfact_dist ${cfact_dist} --use_preds ${use_preds}"

# Execute the command
python $command