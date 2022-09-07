import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

n = 250
d = 500
n_correlated = 25

X_DGP = sample_block_cor_X
X_PARAMS_DICT = {
    "n": n,
    "d": d,
    "rho": [0.8] + [0 for i in range(int(d / n_correlated - 1))],
    "n_blocks": int(d / n_correlated)
}
Y_DGP = logistic_model
Y_PARAMS_DICT = {
    "beta": 1,
    "s": 5
}

VARY_PARAM_NAME = ["beta", "rho"]
VARY_PARAM_VALS = {"beta": {"0.1": 0.1, "1": 1},
                   "rho": {"0.5": [0.5] + [0 for i in range(int(d / n_correlated - 1))], "0.8": [0.8] + [0 for i in range(int(d / n_correlated - 1))], "0.9": [0.9] + [0 for i in range(int(d / n_correlated - 1))], "0.99": [0.99] + [0 for i in range(int(d / n_correlated - 1))]}}

