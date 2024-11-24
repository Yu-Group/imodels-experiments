import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = sample_real_data_X
X_PARAMS_DICT = {
    "source": "imodels",
    "data_name": "csi_pecarn_pred",
    "sample_row_n": None
}
# X_PARAMS_DICT = {
#     "source": "imodels",
#     "data_name": "juvenile",
#     "sample_row_n": None
# }

# X_PARAMS_DICT = {
#     "source": "csv",
#     "file_path": "/accoutns/projects/binyu/zhongyuan_liang/local_MDI+/imodels-experiments/feature_importance/data/Enhancer/X_enhancer_cleaned.csv",
#     "sample_row_n": 2000,
#     "normalize": False
# }

Y_DGP = sample_real_data_y
Y_PARAMS_DICT = {
    "source": "imodels",
    "data_name": "csi_pecarn_pred"
}
# Y_PARAMS_DICT = {
#     "source": "imodels",
#     "data_name": "juvenile"
# }

# Y_PARAMS_DICT = {
#     "source": "csv",
#     "file_path": "/accounts/projects/binyu/zhongyuan_liang/local_MDI+/imodels-experiments/feature_importance/data/Enhancer/y_enhancer.csv",
#     "sample_row_n": 2000
# }


# vary one parameter
VARY_PARAM_NAME = "sample_row_n"
VARY_PARAM_VALS = {"keep_all_rows": None}