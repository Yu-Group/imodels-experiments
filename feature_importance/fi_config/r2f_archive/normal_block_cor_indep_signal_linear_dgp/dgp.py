import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_X
X_PARAMS_DICT = {
    "support": [0, 10, 11, 12, 13],
    "X_fun": sample_block_cor_X,
    "n": 250,
    "d": 50,
    "rho": [0.8, 0, 0, 0, 0],
    "n_blocks": 5
}
Y_DGP = linear_model
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": None,
    "heritability": 0.4,
    "s": 5
}

# VARY_PARAM_NAME = ["heritability", "rho"]
# VARY_PARAM_VALS = {"heritability": {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
#                    "rho": {"0": [0, 0, 0, 0, 0],
#                    "0.2": [0.2, 0, 0, 0, 0],
#                    "0.4": [0.4, 0, 0, 0, 0],
#                    "0.6": [0.6, 0, 0, 0, 0],
#                    "0.8": [0.8, 0, 0, 0, 0]}}

VARY_PARAM_NAME = ["heritability", "n"]
VARY_PARAM_VALS = {"heritability": {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
                   "n": {"100": 100, "250": 250, "500": 500, "1000": 1000}}

# VARY_PARAM_NAME = ["heritability", "s"]
# VARY_PARAM_VALS = {"heritability": {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
#                    "s": {"2": 2, "5": 5, "10": 10, "25": 25}}

# VARY_PARAM_NAME = ["heritability", "sample_col_n"]
# VARY_PARAM_VALS = {"heritability": {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
#                    "sample_col_n": {"100": 100, "250": 250, "500": 500, "1000": 1000}}

# VARY_PARAM_NAME = "rho"
# VARY_PARAM_VALS = {"0": [0, 0, 0, 0, 0],
#                    "0.2": [0.2, 0, 0, 0, 0],
#                    "0.4": [0.4, 0, 0, 0, 0],
#                    "0.6": [0.6, 0, 0, 0, 0],
#                    "0.8": [0.8, 0, 0, 0, 0]}
