import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "data/X_enhancer_uncorrelated_log_transformed.csv",
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
    "frac_label_corruption": None
}

VARY_PARAM_NAME = ["frac_label_corruption", "sample_row_n"]
VARY_PARAM_VALS = {"frac_label_corruption": {"0.0": 0.0, "0.05": 0.05, "0.1": 0.15, "0.25": 0.25,"0.4":0.4},
                   "sample_row_n": {"100": 100, "250": 250, "500": 500,"1000":1000,"1500":1500}}
