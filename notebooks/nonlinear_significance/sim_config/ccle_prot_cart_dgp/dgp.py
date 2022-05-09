# Example usage: run in command line
# cd notebooks/nonlinear_significance
# python 01_run_simulations.py --nreps 2 --config test --split_seed 331 --ignore_cache
# python 01_run_simulations.py --nreps 2 --config enhancer_linear_dgp --split_seed 331 --ignore_cache --create_rmd --show_vars 50

import sys
sys.path.append("../..")
from simulations_util import *

import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


y = pd.read_csv("data/Y_ccle_prot.csv").to_numpy()[:, 1]

X_DGP = sample_model_X
X_PARAMS_DICT = {
    "X_fun": sample_real_X,
    "X_params_dict": {"fpath": "data/X_ccle_prot.csv", "sample_row_n": None, "sample_col_n": None},
    "y": y,
    "model": DecisionTreeRegressor(),
    "n": None
}
Y_DGP = model_based_y
Y_PARAMS_DICT = {
    "y": y,
    "model": DecisionTreeRegressor(),
    "sigma": None,
    "heritability": 0.4,
    "s": 5
}

VARY_PARAM_NAME = ["heritability", "n"]
VARY_PARAM_VALS = {"heritability": {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
                   "n": {"100": 100, "250": 250, "370": 370}}

# VARY_PARAM_NAME = ["heritability", "s"]
# VARY_PARAM_VALS = {"heritability": {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
#                    "s": {"2": 2, "5": 5, "10": 10, "25": 25}}
