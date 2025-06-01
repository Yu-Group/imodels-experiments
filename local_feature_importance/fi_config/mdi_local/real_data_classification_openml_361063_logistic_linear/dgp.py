import sys
sys.path.append("../..")
from local_feature_importance.scripts.simulations_util import *


X_DGP = sample_real_data_X
X_PARAMS_DICT = {
    "task_id": 361063,
    "sample_row_n": None,
    "normalize": True
}


Y_DGP = logistic_linear_model_random_feature
Y_PARAMS_DICT = {
    "beta": 1,
    "s": 5,
    "frac_label_corruption": 0.10
}

VARY_PARAM_NAME = ["frac_label_corruption", "sample_row_n"]
VARY_PARAM_VALS = {"frac_label_corruption": {"0.15": 0.15, "0.10": 0.10, "0.05": 0.05, "0": 0},
                    "sample_row_n": {"150": 150, "500": 500, "1000":1000}}