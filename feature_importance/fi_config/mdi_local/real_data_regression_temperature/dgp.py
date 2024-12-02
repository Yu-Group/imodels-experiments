import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = sample_real_data_X
X_PARAMS_DICT = {
    "source": "uci",
    "data_id": 925,
    "sample_row_n": None
}

Y_DGP = sample_real_data_y
Y_PARAMS_DICT = {
    "source": "uci",
    "data_id": 925
}

VARY_PARAM_NAME = "sample_row_n"
VARY_PARAM_VALS = {"keep_all_rows": None}