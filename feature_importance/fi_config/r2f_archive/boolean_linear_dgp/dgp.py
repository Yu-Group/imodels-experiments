import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_boolean_X
X_PARAMS_DICT = {
    "n": 250,
    "d": 50
}
Y_DGP = linear_model
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": None,
    "heritability": 0.4,
    "s": 5
}

VARY_PARAM_NAME = ["heritability", "n"]
VARY_PARAM_VALS = {"heritability": {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
                   "n": {"100": 100, "250": 250, "500": 500, "1000": 1000}}

# VARY_PARAM_NAME = ["heritability", "s"]
# VARY_PARAM_VALS = {"heritability": {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
#                    "s": {"2": 2, "5": 5, "10": 10, "25": 25}}

# VARY_PARAM_NAME = ["heritability", "sample_col_n"]
# VARY_PARAM_VALS = {"heritability": {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
#                    "sample_col_n": {"100": 100, "250": 250, "500": 500, "1000": 1000}}

# VARY_PARAM_NAME = "max_components"
# VARY_PARAM_VALS = {"placeholder": 0}

# VARY_PARAM_NAME = "n"
# VARY_PARAM_VALS = {'100': 100, '250': 250, '500': 500}

# VARY_PARAM_NAME = "d"
# VARY_PARAM_VALS = {'50': 50, '100': 100, '250': 250}

# VARY_PARAM_NAME = "s"
# VARY_PARAM_VALS = {'5': 5, '10': 10, '15': 15}

# VARY_PARAM_NAME = "sigma"
# VARY_PARAM_VALS = {'0.1': 0.1, '1': 1, '2': 2}
