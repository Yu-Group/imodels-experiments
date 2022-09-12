import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

# linear dgp with s=6 consisting of 3 common and 3 less common signal features

X_DGP = sample_1000g_X
X_PARAMS_DICT = {
    "fpath": "/global/scratch/users/tiffanytang/feature_importance/data/X_1000g_p500.csv",
    "sample_row_n": 1000,
    "sample_col_n": None,
    "snp_order": ["rare", "rare", "rare", "common", "common", "common"]
}
Y_DGP = linear_model
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": None,
    "heritability": 0.4,
    "s": 6
}

VARY_PARAM_NAME = ["heritability", "sample_row_n"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.6": 0.6, "0.8": 0.8},
                   "sample_row_n": {"250": 250, "500": 500, "1000": 1000}}
