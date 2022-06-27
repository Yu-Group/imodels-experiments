import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

y = pd.read_csv("data/Y_tcga.csv").to_numpy().ravel()

X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "data/X_tcga_var_filtered_log_transformed.csv",
    "sample_row_n": None,
    "sample_col_n": None
}
Y_DGP = sample_real_y
Y_PARAMS_DICT = {
    "y": y,
    "s": 5
}

# vary one parameter
VARY_PARAM_NAME = "sample_col_n"
VARY_PARAM_VALS = {"100": 100, "250": 250, "500": 500, "1000": 1000}

# # vary two parameters in a grid
# VARY_PARAM_NAME = ["heritability", "sample_row_n"]
# VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
#                    "sample_row_n": {"100": 100, "250": 250, "500": 500, "1000": 1000}}

# # vary over n_estimators in RF model in models.py
# VARY_PARAM_NAME = "n_estimators"
# VARY_PARAM_VALS = {"placeholder": 0}