import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = sample_real_data_X
X_PARAMS_DICT = {
    "source": "imodels",
    "data_name": "diabetes_regr",
    "sample_row_n": None
}
# X_PARAMS_DICT = {
#     "source": "imodels",
#     "data_name": "satellite_image",
#     "sample_row_n": None
# }
# X_PARAMS_DICT = {
#     "source": "openml",
#     "task_id": 359946,
#     "sample_row_n": None
# }

Y_DGP = sample_real_data_y
Y_PARAMS_DICT = {
    "source": "imodels",
    "data_name": "diabetes_regr"
}
# Y_PARAMS_DICT = {
#     "source": "imodels",
#     "data_name": "satellite_image"
# }
# Y_PARAMS_DICT = {
#     "source": "openml",
#     "task_id": 359946
# }

# vary one parameter
VARY_PARAM_NAME = "sample_row_n"
VARY_PARAM_VALS = {"keep_all_rows": None}