import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "data/X_enhancer_uncorrelated_log_transformed.csv",
    "sample_row_n": None,
    "sample_col_n": None
}
Y_DGP = partial_linear_lss_model
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": None,
    "heritability": 0.4,
    "tau": 0,
    "m": 3,
    "r": 2,
    "s": 1,
}

VARY_PARAM_NAME = ["heritability", "sample_row_n"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
                   "sample_row_n": {"100": 100, "250": 250, "500": 500,"1000":1000,"1500":1500}}
