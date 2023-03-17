import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = entropy_X
X_PARAMS_DICT = {
    "n": 100
}
Y_DGP = entropy_y
Y_PARAMS_DICT = {
    "c": 3
}

VARY_PARAM_NAME = ["c", "n"]
VARY_PARAM_VALS = {"c": {"3": 3},
                   "n": {"50": 50, "100": 100, "250": 250, "500": 500, "1000": 1000}}
