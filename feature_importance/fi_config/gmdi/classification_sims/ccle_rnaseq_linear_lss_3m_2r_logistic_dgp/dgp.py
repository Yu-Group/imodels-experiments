import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "/global/scratch/users/tiffanytang/feature_importance/data/X_ccle_rnaseq_cleaned.csv",
    "sample_row_n": None,
    "sample_col_n": 1000
}
Y_DGP = logistic_partial_linear_lss_model
Y_PARAMS_DICT = {
    "s":1,
    "m":3,
    "r":2,
    "tau":0,
    "beta": 1,
    "frac_label_corruption": None
}

VARY_PARAM_NAME = ["frac_label_corruption", "sample_row_n"]
VARY_PARAM_VALS = {"frac_label_corruption": {"0.25": 0.25, "0.15": 0.15, "0.05": 0.05, "0": None},
                   "sample_row_n": {"100": 100, "250": 250, "472": 472}}
