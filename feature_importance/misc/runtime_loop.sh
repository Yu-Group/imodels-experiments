#!/bin/bash

# define bash file
slurm_script="runtime.sh"

# define arrays for n and p
# n_values=(1000 10000 50000 100000 150000 174368)
n_values=(1000 10000 50000)
p_values=(10 50 100 200 500 1000 2000 5000 10000)
#lfi_methods=("linear_partial" "r2")
lfi_methods=("linear_partial")

# iterate over two reps
for rep in {1..2}; do
  # iterate over each lfi method
  for lfi_method in "${lfi_methods[@]}"; do
    # iterate over each value of n
    for n in "${n_values[@]}"; do
      # iterate over each value of p
      for p in "${p_values[@]}"; do
        # check if n is greater than p
        if [ "$n" -gt "$p" ]; then
          # run the bash file with the current combination of n and p
          sbatch $slurm_script $n $p $lfi_method $rep
        fi
      done
    done
  done
done