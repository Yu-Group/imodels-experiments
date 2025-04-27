import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = sample_real_data_X
X_PARAMS_DICT = {
    "task_id": 361260,
    "sample_row_n": None,
    "normalize": True
}


Y_DGP = linear_model_random_feature
Y_PARAMS_DICT = {
    "beta": 1,
    "heritability": 0.4,
    "s": 5,
}

VARY_PARAM_NAME = ["heritability", "sample_row_n"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
                   "sample_row_n": {"150": 150, "500": 500, "1000":1000}}