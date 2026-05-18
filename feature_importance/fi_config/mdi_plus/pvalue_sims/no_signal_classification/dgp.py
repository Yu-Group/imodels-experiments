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
    "beta": 0,
    "s": 0
}

VARY_PARAM_NAME = ["sigma", "n"]
VARY_PARAM_VALS = {"d": {"10": 10, "100": 100},
                   "n": {"100": 100, "250": 250, "500": 500, "1000": 1000}}
