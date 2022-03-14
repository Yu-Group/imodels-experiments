import sys
sys.path.append("../..")
from simulations_util import *

X_DGP = sample_normal_X
X_PARAMS_DICT = {
    "n": 100,
    "d": 50
}
Y_DGP = linear_model
Y_PARAMS_DICT = {
    "beta": 1,
    "sigma": 0.1,
    "s": 5
}
VARY_PARAM_NAME = "n"
VARY_PARAM_VALS = {'100': 100, '250': 250, '500': 500}