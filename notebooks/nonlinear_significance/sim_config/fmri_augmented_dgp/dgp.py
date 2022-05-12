# Example usage: run in command line
# cd notebooks/nonlinear_significance
# python 01_run_simulations.py --nreps 2 --config test --split_seed 331 --ignore_cache
# python 01_run_simulations.py --nreps 2 --config enhancer_linear_dgp --split_seed 331 --ignore_cache --create_rmd --show_vars 50

import sys
sys.path.append("../..")
from simulations_util import *

import pandas as pd

y = pd.read_csv("data/Y_fmri.csv").to_numpy()[:, 0]

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "/global/scratch/users/tiffanytang/nonlinear-significance/data/X_fmri.csv",
    "sample_row_n": 1000,
    "sample_col_n": 500,
    "signal_features": ["V4940", "V4925", "V5176", "V4680", "V4112",
                        "V1101", "V3827", "V1392", "V5108", "V4376"],
    "n_signal_features": 10,
    "permute_nonsignal_col": "augment"
}
Y_DGP = sample_real_y
Y_PARAMS_DICT = {
    "y": y,
    "s": 10
}

VARY_PARAM_NAME = ["sample_col_n", "sample_row_n"]
VARY_PARAM_VALS = {"sample_col_n": {"100": 100, "250": 250, "500": 500, "1000": 1000},
                   "sample_row_n": {"100": 100, "250": 250, "500": 500, "1000": 1000}}

# VARY_PARAM_NAME = ["heritability", "s"]
# VARY_PARAM_VALS = {"heritability": {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
#                    "s": {"2": 2, "5": 5, "10": 10, "25": 25}}

# VARY_PARAM_NAME = ["heritability", "sample_col_n"]
# VARY_PARAM_VALS = {"heritability": {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
#                    "sample_col_n": {"100": 100, "250": 250, "500": 500, "1000": 1000}}
