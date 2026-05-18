import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_normal_X
X_PARAMS_DICT = {
    "n": 500,
    "d": 100
}
Y_DGP = logistic_model
Y_PARAMS_DICT = {
    "beta": 3,
    "s": 2,
    "frac_label_corruption": None
}

VARY_PARAM_NAME = ["frac_label_corruption", "n"]
VARY_PARAM_VALS = {"frac_label_corruption": {"0.25": 0.25, "0.15": 0.15, "0.05": 0.05, "0": None},
                   "n": {"100": 100, "250": 250, "500": 500, "1000": 1000}}
