#!/bin/bash

command="00_run_stability.py --nreps 1 --config mdi_local.real_data_${1}_${2}_stability --split_seed ${3} --sample_seed ${4} --ignore_cache --create_rmd --folder_name ${2}_stability"

python $command