#!/bin/bash
#SBATCH --mail-user=zachrewolinski@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=24


source activate mdi
# command="runtime_test.py --n_samples 40000 --n_features 10000 --lfi_method linear_partial --rep 1"
command="runtime_test.py --n_samples ${1} --n_features ${2} --lfi_method ${3} --rep ${4}"

# Execute the command
python $command