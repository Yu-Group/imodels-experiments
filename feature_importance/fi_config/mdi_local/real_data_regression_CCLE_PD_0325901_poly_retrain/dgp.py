import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *


X_DGP = sample_real_data_X
X_PARAMS_DICT = {
    "source": "csv",
    "file_path": "/accounts/projects/binyu/zhongyuan_liang/local_MDI+/imodels-experiments/feature_importance/data/CCLE/X_ccle_rnaseq_PD-0325901_top500.csv",
    "sample_row_n": None
}
# X_PARAMS_DICT = {
#     "source": "imodels",
#     "data_name": "satellite_image",
#     "sample_row_n": None
# }
# X_PARAMS_DICT = {
#     "source": "openml",
#     "data_id": 588,
#     "sample_row_n": None
# }
# X_PARAMS_DICT = {
#     "source": "csv",
#     "file_path": "/accounts/projects/binyu/zhongyuan_liang/local_MDI+/imodels-experiments/feature_importance/data/CCLE/X_ccle_rnaseq_PD-0325901_top1000.csv",
#     "sample_row_n": None
# }

Y_DGP = hierarchical_poly
Y_PARAMS_DICT = {
    "m": 3,
    "r": 2,
    "beta": 1,
    "heritability": 0.4,
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
VARY_PARAM_NAME = ["heritability"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8}}