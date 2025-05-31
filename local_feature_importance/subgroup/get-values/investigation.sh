#!/bin/bash

source activate mdi
command="investigation.py --dataname ${1} --seed ${2} --method ${3}"

# Execute the command
python $command