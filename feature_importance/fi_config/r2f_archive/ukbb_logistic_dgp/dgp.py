import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "data/X_ukbb_predixcan_lv_small.csv",
    "sample_row_n": 1000,
    "sample_col_n": 500
}
Y_DGP = logistic_model
Y_PARAMS_DICT = {
    "beta": 1,
    "s": 5
}

VARY_PARAM_NAME = "sample_row_n"
VARY_PARAM_VALS = {"100": 100, "250": 250, "500": 500, "1000": 1000}
