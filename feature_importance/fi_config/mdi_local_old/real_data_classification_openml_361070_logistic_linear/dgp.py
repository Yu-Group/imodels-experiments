import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = sample_real_data_X
X_PARAMS_DICT = {
    "source": "openml",
    "task_id": 361070,
    "sample_row_n": None,
    "normalize": True
}


Y_DGP = logistic_linear_model_random_feature
Y_PARAMS_DICT = {
    "beta": 1,
    "s": 5,
    "frac_label_corruption": 0.10
}
# VARY_PARAM_NAME = ["frac_label_corruption"]
# VARY_PARAM_VALS = {"frac_label_corruption": {"0.15": 0.15, "0.10": 0.10, "0.05": 0.05, "0": 0}}

# VARY_PARAM_NAME = ["frac_label_corruption", "sample_row_n"]
# VARY_PARAM_VALS = {"frac_label_corruption": {"0.15": 0.15, "0.10": 0.10, "0.05": 0.05, "0": 0},
#                    "sample_row_n": {"150": 150, "300": 300, "500": 500}}
# VARY_PARAM_NAME = ["frac_label_corruption", "sample_row_n"]
# VARY_PARAM_VALS = {"frac_label_corruption": {"0.15": 0.15, "0.10": 0.10, "0.05": 0.05, "0": 0},
#                    "sample_row_n": {"2000": 2000}}
VARY_PARAM_NAME = ["frac_label_corruption", "sample_row_n"]
VARY_PARAM_VALS = {"frac_label_corruption": {"0.15": 0.15, "0.10": 0.10, "0.05": 0.05, "0": 0},
                   "sample_row_n": {"150": 150, "300": 300, "500": 500}}