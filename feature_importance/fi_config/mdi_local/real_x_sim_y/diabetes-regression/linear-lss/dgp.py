import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "../data/regression_data/Diabetes_regression/X_diabetes_regression.csv",
    "sample_row_n": 442
}
Y_DGP = hierarchical_poly
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": None,
    "heritability": 0.4,
    "m": 3,
    "r": 2
}

VARY_PARAM_NAME = ["heritability", "sample_row_n"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2,
                                    "0.4": 0.4, "0.8": 0.8},
                   "sample_row_n": {"100": 100, "200": 200,
                                    "300": 300, "400": 400}}