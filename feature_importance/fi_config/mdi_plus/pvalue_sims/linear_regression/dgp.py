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
    "beta": 1,
    "s": 2,
    "sigma": None,
    "heritability": 0.4
}

VARY_PARAM_NAME = ["heritability", "n"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
                   "n": {"100": 100, "250": 250, "500": 500, "1000": 1000}}
