import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "/global/scratch/users/tiffanytang/feature_importance/data/X_juvenile_cleaned.csv",
    "sample_row_n": 1000,
    "sample_col_n": None
}
Y_DGP = linear_model
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": None,
    "heritability": 0.4,
    "s":5,
}

VARY_PARAM_NAME = ["heritability", "s"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
                   "s": {"1": 1, "5": 5, "10": 10, "25": 25, "50": 50}}
