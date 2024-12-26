import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = sample_real_data_X
X_PARAMS_DICT = {
    "source": "openml",
    "task_id": 9946,
    "sample_row_n": None
}


Y_DGP = sample_real_data_y
Y_PARAMS_DICT = {
    "source": "openml",
    "task_id": 9946
}
# Y_PARAMS_DICT = {
#     "source": "imodels",
#     "data_name": "satellite_image"
# }
# Y_PARAMS_DICT = {
#     "source": "openml",
#     "data_id": 588
# }

# Y_PARAMS_DICT = {
#     "source": "csv",
#     "file_path": "/accounts/projects/binyu/zhongyuan_liang/local_MDI+/imodels-experiments/feature_importance/data/CCLE/y_ccle_rnaseq_PD-0325901.csv",
# }

# vary one parameter
VARY_PARAM_NAME = "sample_row_n"
VARY_PARAM_VALS = {"keep_all_rows": None}