import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = sample_real_X
X_PARAMS_DICT = {
    "fpath": "../data/regression_data/Diabetes_regression/X_diabetes_regression.csv",
    "sample_row_n": 442
}
# X_PARAMS_DICT = {
#     "X_fpath": "../data/classification_data/Fico/X_fico.csv",
#     "sample_row_n": None,
#     "return_data": "X"
# }
# X_PARAMS_DICT = {
#     "X_fpath": "../data/classification_data/Juvenile/X_juvenile.csv",
#     "sample_row_n": None,
#     "return_data": "X"
# }
Y_DGP = linear_model
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": None,
    "heritability": 0.4,
    "s": 5
}
# Y_PARAMS_DICT = {
#     "y_fpath": "../data/classification_data/Fico/y_fico.csv",
#     "return_data": "y"
# }
# Y_PARAMS_DICT = {
#     "y_fpath": "../data/classification_data/Juvenile/y_juvenile.csv",
#     "return_data": "y"
# }

# vary one parameter
VARY_PARAM_NAME = ["heritability", "sample_row_n"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2,
                                    "0.4": 0.4, "0.8": 0.8},
                   "sample_row_n": {"100": 100, "200": 200,
                                    "300": 300, "400": 400}}

# VARY_PARAM_NAME = ["heritability"]
# VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2,
#                                     "0.4": 0.4, "0.8": 0.8}}
