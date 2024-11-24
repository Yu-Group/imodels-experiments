import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


### Update start for local MDI+
X_DGP = sample_real_data_X
X_PARAMS_DICT = {
    "source": "uci",
    "data_id": 183,
    "sample_row_n": None
}
Y_DGP = sample_real_data_y
Y_PARAMS_DICT = {
    "source": "uci",
    "data_id": 183
}
### Update for local MDI+ done

# # vary one parameter
# VARY_PARAM_NAME = "sample_row_n"
# VARY_PARAM_VALS = {"100": 100, "250": 250, "500": 500, "1000": 1000}

# vary two parameters in a grid
VARY_PARAM_NAME = "sample_row_n"
VARY_PARAM_VALS = {"keep_all_rows": None}