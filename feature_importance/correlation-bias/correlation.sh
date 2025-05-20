#!/bin/bash

njobs=8

source activate mdi
command="correlation_pipeline.py --seed ${1} --pve ${2} --rho ${3} --njobs $njobs"

# Execute the command
python $command