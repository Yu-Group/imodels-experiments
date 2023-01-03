import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "/global/scratch/users/tiffanytang/feature_importance/data/X_ccle_rnaseq_cleaned.csv",
    "sample_row_n": None,
    "sample_col_n": 1000
}
Y_DGP = lss_model
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": None,
    "heritability": 0.4,
    "tau": 0,
    "m": 3,
    "r": 2
}

VARY_PARAM_NAME = ["heritability", "sample_col_n"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
                   "sample_col_n": {"10": 10, "25": 25, "50": 50, "100": 100, "250": 250, "500": 500, "1000": 1000, "2000": 2000}}
