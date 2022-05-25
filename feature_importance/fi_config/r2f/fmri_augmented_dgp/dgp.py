import pandas as pd
import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

y = pd.read_csv("data/Y_fmri.csv").to_numpy()[:, 0]

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "data/X_fmri.csv",
    "sample_row_n": 1000,
    "sample_col_n": 500,
    "signal_features": ["V4940", "V4925", "V5176", "V4680", "V4112",
                        "V1101", "V3827", "V1392", "V5108", "V4376"],
    "n_signal_features": 10,
    "permute_nonsignal_col": "augment"
}
Y_DGP = sample_real_y
Y_PARAMS_DICT = {
    "y": y,
    "s": 10
}

VARY_PARAM_NAME = ["sample_col_n", "sample_row_n"]
VARY_PARAM_VALS = {"sample_col_n": {"100": 100, "250": 250, "500": 500, "1000": 1000},
                   "sample_row_n": {"100": 100, "250": 250, "500": 500, "1000": 1000}}
