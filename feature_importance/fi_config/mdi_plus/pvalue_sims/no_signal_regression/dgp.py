import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_normal_X
X_PARAMS_DICT = {
    "n": 500,
    "d": 100
}
Y_DGP = linear_model
Y_PARAMS_DICT = {
    "beta": 0,
    "sigma": 1,
    "s": 0
}

VARY_PARAM_NAME = ["sigma", "n"]
VARY_PARAM_VALS = {"sigma": {"0.5": 0.5, "1.0": 1.0, "2.0": 2.0},
                   "n": {"100": 100, "250": 250, "500": 500, "1000": 1000}}
