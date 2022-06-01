import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "data/X_ccle_prot.csv",
    "sample_row_n": None,
    "sample_col_n": None,
    "permute_nonsignal_col": "indep"
}
Y_DGP = logistic_model
Y_PARAMS_DICT = {
    "beta": 1,
    "s": 5
}

VARY_PARAM_NAME = "sample_row_n"
VARY_PARAM_VALS = {"100": 100, "250": 250, "370": 370}
