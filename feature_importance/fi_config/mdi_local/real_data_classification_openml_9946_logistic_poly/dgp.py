import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = sample_real_data_X
X_PARAMS_DICT = {
    "source": "openml",
    "task_id": 9946,
    "sample_row_n": None,
    "normalize": True
}


Y_DGP = logistic_poly_random_feature
Y_PARAMS_DICT = {
    "m": 3,
    "r": 2,
    "beta": 1,
    "frac_label_corruption": 0.25
}
VARY_PARAM_NAME = ["frac_label_corruption"]
VARY_PARAM_VALS = {"frac_label_corruption": {"0.25": 0.25, "0.20": 0.20, "0.10": 0.10, "0.05": 0.05}}