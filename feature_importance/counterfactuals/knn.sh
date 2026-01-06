#!/bin/bash

src="openml"

source activate mdi
command="knn.py --datasource ${src} --dataid ${1} --k ${2} --nbr_dist ${3} --cfact_dist ${4} --use_preds ${5}"

# execute the command
python $command