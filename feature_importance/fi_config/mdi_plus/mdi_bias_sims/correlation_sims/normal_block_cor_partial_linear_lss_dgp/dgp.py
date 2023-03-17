import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

n = 250
d = 100
n_correlated = 50

X_DGP = sample_block_cor_X
X_PARAMS_DICT = {
    "n": n,
    "d": d,
    "rho": [0.8] + [0 for i in range(int(d / n_correlated - 1))],
    "n_blocks": int(d / n_correlated)
}
Y_DGP = partial_linear_lss_model
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": None,
    "heritability": 0.4,
    "s": 1,
    "m": 3,
    "r": 2,
    "tau": 0
}

VARY_PARAM_NAME = ["heritability", "rho"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.4": 0.4},
                   "rho": {"0.5": [0.5] + [0 for i in range(int(d / n_correlated - 1))], "0.6": [0.6] + [0 for i in range(int(d / n_correlated - 1))], "0.7": [0.7] + [0 for i in range(int(d / n_correlated - 1))], "0.8": [0.8] + [0 for i in range(int(d / n_correlated - 1))], "0.9": [0.9] + [0 for i in range(int(d / n_correlated - 1))], "0.99": [0.99] + [0 for i in range(int(d / n_correlated - 1))]}}
