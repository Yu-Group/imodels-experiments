import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "data/X_juvenile.csv",
    "sample_row_n": None,
    "sample_col_n": None
}
Y_DGP = logistic_lss_model
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": None,
    "m":3,
    "r":2,
    "tau":0,
}

VARY_PARAM_NAME = ["beta", "sample_row_n"]
VARY_PARAM_VALS = {"beta": {"0.1": 0.1, "0.25": 0.25, "0.5": 0.5, "1.0": 1.0},
                   "sample_row_n": {"100": 100, "250": 250, "500": 500,"1000":1000,"1500":1500}}
