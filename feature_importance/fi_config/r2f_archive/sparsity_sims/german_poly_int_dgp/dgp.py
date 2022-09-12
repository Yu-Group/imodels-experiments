import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "data/X_german.csv",
    "sample_row_n": None,
    "sample_col_n": None
}
Y_DGP = hierarchical_poly
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": None,
    "heritability": 0.4,
    "m": 2,
    "r": 3
}

VARY_PARAM_NAME = ["heritability", "m"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
                   "m": {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}}
