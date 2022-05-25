import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "data/X_german.csv",
    "sample_row_n": None,
    "sample_col_n": None
}
Y_DGP = linear_model
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": None,
    "heritability": 0.4,
    "s": 5
}

VARY_PARAM_NAME = ["heritability", "s"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
                   "s": {"2": 2, "5": 5, "10": 10, "15": 15}}
