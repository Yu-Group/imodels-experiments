import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = sample_real_data_X
X_PARAMS_DICT = {
    "task_id": 361243,
    "sample_row_n": None
}


Y_DGP = sample_real_data_y
Y_PARAMS_DICT = {
    "task_id": 361243
}

# vary one parameter
VARY_PARAM_NAME = ["sample_row_n"]
VARY_PARAM_VALS = {"sample_row_n":{"10000":10000}}
# VARY_PARAM_VALS = {"sample_row_n":{"500":500, "1000":1000, "2000":2000}}