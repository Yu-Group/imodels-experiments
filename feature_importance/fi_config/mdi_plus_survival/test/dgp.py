import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

import sksurv
from sksurv import datasets


X_DGP = load_X_data
X_PARAMS_DICT = {
    "data_fn": sksurv.datasets.load_aids
}
Y_DGP = load_y_data
Y_PARAMS_DICT = {
    "data_fn": sksurv.datasets.load_aids
}

VARY_PARAM_NAME = None
VARY_PARAM_VALS = None
# VARY_PARAM_NAME = ["corrupt_size", "sample_row_n"]
# VARY_PARAM_VALS = {"corrupt_size": {"0": 0, "0.01": 0.005, "0.025": 0.0125, "0.05": 0.025},
#                    "sample_row_n": {"100": 100, "250": 250, "472": 472}}
