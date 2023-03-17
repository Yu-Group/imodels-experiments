import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "data/X_ccle_rnaseq_cleaned.csv",
    "sample_row_n": None,
    "sample_col_n": 1000
}
Y_DGP = hierarchical_poly
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": None,
    "heritability": 0.4,
    "m": 3,
    "r": 2,
    "corrupt_how": "leverage_normal",
    "corrupt_size": 0.1,
    "corrupt_mean": 25
}

VARY_PARAM_NAME = ["corrupt_size", "sample_row_n"]
VARY_PARAM_VALS = {"corrupt_size": {"0": 0, "0.01": 0.005, "0.025": 0.0125, "0.05": 0.025},
                   "sample_row_n": {"100": 100, "250": 250, "472": 472}}
