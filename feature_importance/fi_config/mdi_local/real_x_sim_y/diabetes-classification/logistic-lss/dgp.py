import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "../data/classification_data/Diabetes/X_diabetes.csv",
    "sample_row_n": 768
}

Y_DGP = logistic_partial_linear_lss_model
Y_PARAMS_DICT = {
    "s":1,
    "m":3,
    "r":2,
    "tau":0,
    "beta": 1,
    "heritability": 0.4
}
VARY_PARAM_NAME = ["heritability", "sample_row_n"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2,
                                    "0.4": 0.4, "0.8": 0.8},
                   "sample_row_n": {"100": 100, "200": 200,
                                    "400": 400, "600": 600}}
                   
