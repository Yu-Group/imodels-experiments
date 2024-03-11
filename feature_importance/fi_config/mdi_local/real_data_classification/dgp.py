import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = sample_real_data
X_PARAMS_DICT = {
    "X_fpath": "../data/classification_data/Diabetes/X_diabetes.csv",
    "sample_row_n": None,
    "return_data": "X"
}
Y_DGP = sample_real_data
Y_PARAMS_DICT = {
    "y_fpath": "../data/classification_data/Diabetes/y_diabetes.csv",
    "return_data": "y"
}

# vary one parameter
VARY_PARAM_NAME = "sample_row_n"
VARY_PARAM_VALS = {"keep_all_rows": None}