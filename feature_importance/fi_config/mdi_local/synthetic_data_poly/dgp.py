import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

### Update start for local MDI+
X_DGP = sample_normal_X_subgroups
X_PARAMS_DICT = {
    "n": 1000,
    "d": 10,
    "mean": [[0]*10,[0]*5+[0]*5],
    "scale": [[1]*10,[1]*10],
}
Y_DGP = hierarchical_poly
Y_PARAMS_DICT = {
    "m": 3,
    "sigma": None,
    "heritability": 0.4,
    "r": 2,
}
### Update for local MDI+ done

# # vary one parameter
# VARY_PARAM_NAME = "sample_row_n"
# VARY_PARAM_VALS = {"100": 100, "250": 250, "500": 500, "1000": 1000}

# vary two parameters in a grid
VARY_PARAM_NAME = ["heritability", "n"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.2": 0.2, "0.4": 0.4, "0.8": 0.8},
                   "n": {"100": 100, "250": 250, "500": 500, "1000": 1000}}

# # vary over n_estimators in RF model in models.py
# VARY_PARAM_NAME = "n_estimators"
# VARY_PARAM_VALS = {"placeholder": 0}