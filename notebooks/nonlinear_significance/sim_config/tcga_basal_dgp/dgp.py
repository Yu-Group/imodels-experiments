# Example usage: run in command line
# cd notebooks/nonlinear_significance
# python 01_run_simulations.py --nreps 2 --config test --split_seed 331 --ignore_cache
# python 01_run_simulations.py --nreps 2 --config enhancer_linear_dgp --split_seed 331 --ignore_cache --create_rmd --show_vars 50

import sys
sys.path.append("../..")
from simulations_util import *

import pandas as pd

y = pd.read_csv("data/Y_tcga.csv").to_numpy()[:, 0]
y = (y == "Basal") * 1

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "data/X_tcga_var_filtered_log_transformed.csv",
    "sample_row_n": None,
    "sample_col_n": None,
    "permute_col": False,
    "normalize": True
}
Y_DGP = sample_real_y
Y_PARAMS_DICT = {
    "y": y,
    "s": 1. # dummy
}

VARY_PARAM_NAME = "s"
VARY_PARAM_VALS = {"1": 1}
