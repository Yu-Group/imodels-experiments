# Example usage: run in command line
# cd notebooks/nonlinear_significance
# python 01_run_simulations.py --nreps 2 --config test --split_seed 331 --ignore_cache
# python 01_run_simulations.py --nreps 2 --config test --split_seed 331 --ignore_cache --create_rmd

import sys
sys.path.append("../..")
from simulations_util import *

X_DGP = sample_normal_X
X_PARAMS_DICT = {
    "n": 250,
    "d": 50
}
Y_DGP = lss_model
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": 0.1,
    "tau": 0,
    "m": 3,
    "r": 2
}

VARY_PARAM_NAME = "n"
VARY_PARAM_VALS = {'100': 100, '250': 250, '500': 500}

# VARY_PARAM_NAME = "m"
# VARY_PARAM_VALS = {"1": 1, "3": 3, "5": 5}

# VARY_PARAM_NAME = "sigma"
# VARY_PARAM_VALS = {"0.1": 0.1, "1": 1, "2": 2}