import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "../data/regression_data/CA_housing/X_california_housing.csv",
    "sample_row_n": None
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

VARY_PARAM_NAME = ["heritability", "sample_row_n"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2,
                                    "0.4": 0.4, "0.8": 0.8},
                   "sample_row_n": {"500": 500, "1000": 1000,
                                    "2000": 2000, "4000": 4000,
                                    "6000": 6000, "8000": 8000,
                                    "10000": 10000}}