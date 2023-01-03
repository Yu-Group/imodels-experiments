import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "/global/scratch/users/tiffanytang/feature_importance/data/X_enhancer_uncorrelated_log_transformed.csv",
    "sample_row_n": None,
    "sample_col_n": None
}
Y_DGP = logistic_model
Y_PARAMS_DICT = {
    "s": 5,
    "beta": 1,
    "frac_label_corruption": None
}

VARY_PARAM_NAME = ["frac_label_corruption", "sample_row_n"]
VARY_PARAM_VALS = {"frac_label_corruption": {"0": None, "0.05": 0.05, "0.15": 0.15, "0.25": 0.25},
                   "sample_row_n": {"100": 100, "250": 250, "500": 500, "1000": 1000, "1500": 1500}}
