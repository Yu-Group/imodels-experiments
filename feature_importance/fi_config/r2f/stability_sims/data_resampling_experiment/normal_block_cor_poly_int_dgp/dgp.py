import sys
sys.path.append("../..")
from feature_importance.scripts.simulations_util import *

X_DGP = sample_block_cor_X
X_PARAMS_DICT = {
    "n": 250,
    "d": 50,
    "rho": 0.8,
    "n_blocks": 2
    "first_block_only": True
}
Y_DGP = hierarchical_poly
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": None,
    "heritability": 0.4,
    "m": 2,
    "r": 3
}

VARY_PARAM_NAME = ["heritability", "rho"]
VARY_PARAM_VALS = {"heritability": {"0.1": 0.1, "0.4": 0.4},
                   "rho": {"0.5": 0.5, "0.8": 0.8, "0.99": 0.99}}