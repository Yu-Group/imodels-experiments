import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = sample_real_data_X
X_PARAMS_DICT = {
    "source": "openml",
    "task_id": 361242,
    "sample_row_n": None,
    "normalize": True
}


Y_DGP = partial_linear_lss_model_random_feature
Y_PARAMS_DICT = {
    "s": 1,
    "m": 3,
    "r": 2,
    "tau": "median",
    "beta": 1,
    "heritability": 0.4,
}

# # vary one parameter
# VARY_PARAM_NAME = ["heritability"]
# VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8}}

# VARY_PARAM_NAME = ["heritability", "sample_row_n"]
# VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
#                    "sample_row_n": {"150": 150, "300": 300, "500": 500}}

# VARY_PARAM_NAME = ["heritability", "sample_row_n"]
# VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
#                    "sample_row_n": {"2000": 2000}}
VARY_PARAM_NAME = ["heritability", "sample_row_n"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
                   "sample_row_n": {"150": 150, "300": 300, "500": 500}}