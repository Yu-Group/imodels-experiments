# Example usage: run in command line
# cd notebooks/nonlinear_significance
# python 01_run_simulations.py --nreps 2 --config test --split_seed 331 --ignore_cache
# python 01_run_simulations.py --nreps 2 --config enhancer_linear_dgp --split_seed 331 --ignore_cache --create_rmd --show_vars 50

import sys
sys.path.append("../..")
from simulations_util import *

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "/global/scratch/users/tiffanytang/nonlinear-significance/data/X_ukbb_predixcan_lv_small.csv",
    "sample_row_n": 1000,
    "sample_col_n": 500
}
Y_DGP = hierarchical_poly
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": None,
    "heritability": 0.4,
    "m": 2,
    "r": 3
}

VARY_PARAM_NAME = ["heritability", "sample_row_n"]
VARY_PARAM_VALS = {"heritability": {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
                   "sample_row_n": {"100": 100, "250": 250, "500": 500, "1000": 1000}}

# VARY_PARAM_NAME = ["heritability", "m"]
# VARY_PARAM_VALS = {"heritability": {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
#                    "m": {"2": 2, "3": 3, "5": 5, "8": 8}}

# VARY_PARAM_NAME = ["heritability", "sample_col_n"]
# VARY_PARAM_VALS = {"heritability": {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
#                    "sample_col_n": {"100": 100, "250": 250, "500": 500, "1000": 1000}}
