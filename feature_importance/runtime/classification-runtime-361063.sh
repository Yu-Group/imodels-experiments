#!/bin/bash
#SBATCH --partition=yugroup

id="361063"
classification=1

source activate mdi
command="runtime_analysis.py --dataid ${id} --classification ${classification}"
python $command
