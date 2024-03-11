import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = sample_real_data
X_PARAMS_DICT = {
    "X_fpath": "../data/regression_data/Diabetes_regression/X_diabetes_regression.csv",
    "sample_row_n": None,
    "return_data": "X"
}
Y_DGP = sample_real_data
Y_PARAMS_DICT = {
    "y_fpath": "../data/regression_data/Diabetes_regression/y_diabetes_regression.csv",
    "return_data": "y"
}

# vary one parameter
VARY_PARAM_NAME = "sample_row_n"
VARY_PARAM_VALS = {"None": None}