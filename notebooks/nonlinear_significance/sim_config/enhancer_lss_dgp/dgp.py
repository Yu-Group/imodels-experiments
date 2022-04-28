# Example usage: run in command line
# cd notebooks/nonlinear_significance
# python 01_run_simulations.py --nreps 2 --config test --split_seed 331 --ignore_cache
# python 01_run_simulations.py --nreps 2 --config enhancer_linear_dgp --split_seed 331 --ignore_cache --create_rmd --show_vars 50

import sys
sys.path.append("../..")
from simulations_util import *

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "data/X_enhancer_uncorrelated.csv",
    "sample_row_n": 1000
}
Y_DGP = lss_model
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": 0.1,
    "tau": 0,
    "m": 3,
    "r": 2
}

#VARY_PARAM_NAME = "n"
#VARY_PARAM_VALS = {'100': 100, '250': 250, '500': 500}

# VARY_PARAM_NAME = "d"
# VARY_PARAM_VALS = {'50': 50, '100': 100, '250': 250}

# VARY_PARAM_NAME = "s"
# VARY_PARAM_VALS = {'5': 5, '10': 10, '15': 15}

VARY_PARAM_NAME = "sigma"
VARY_PARAM_VALS = {'0.1': 0.1, '1': 1, '2': 2, '4': 4}