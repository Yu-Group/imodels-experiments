#!/bin/bash

command="00_run_selection.py --nreps 1 --config mdi_local.real_data_${1}_${2} --split_seed ${3} --sample_seed ${4} --ignore_cache --create_rmd --folder_name ${2}_selection2"

python $command