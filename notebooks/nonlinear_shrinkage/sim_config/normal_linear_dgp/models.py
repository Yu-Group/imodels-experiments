# Note: This is currently just a test file with dummy/placeholder functions

from functools import partial

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from imodels import (
    GreedyTreeClassifier, GreedyTreeRegressor, HSTreeClassifierCV, HSTreeRegressorCV
)
from util import ModelConfig, FIModelConfig
from util_metrics import MDI

ENSEMBLE_ESTIMATOR_NUMS = [3, 10, 25, 50]
TREE_DEPTHS = [1, 2, 3, 4, 5, 7, 8, 10, 15, 20, 25]
ESTIMATORS = [
    [ModelConfig('CART_(MSE)', GreedyTreeRegressor, 'max_depth', n, model_type='tree')
     for n in TREE_DEPTHS],
    # [Model('CART_(MAE)', GreedyTreeRegressor, 'max_depth', n, other_params={'criterion': 'absolute_error'})
    #  for n in TREE_DEPTHS],
    # [ModelConfig('HSCART', partial(HSTreeRegressorCV, estimator_=DecisionTreeRegressor(max_depth=n)))
    #  for n in TREE_DEPTHS],
    # [ModelConfig('Random_Forest', RandomForestRegressor, other_params={'n_estimators': n})
    #  for n in ENSEMBLE_ESTIMATOR_NUMS],
    # [ModelConfig('HSRandom_Forest',
    #              partial(HSTreeRegressorCV, estimator_=RandomForestRegressor(n_estimators=n)))
    #  for n in ENSEMBLE_ESTIMATOR_NUMS],
    # [ModelConfig('Gradient_Boosting', GradientBoostingRegressor, 'n_estimators', n,
    #              other_params=RANDOM_FOREST_DEFAULT_KWARGS)
    #  for n in ENSEMBLE_ESTIMATOR_NUMS],
    # [ModelConfig('HSGradient_Boosting',
    #              partial(HSTreeRegressorCV, estimator_=GradientBoostingRegressor(n_estimators=n)))
    #  for n in ENSEMBLE_ESTIMATOR_NUMS],
]

FI_ESTIMATORS = [
    [FIModelConfig('MDI', MDI, 'train-test', False, model_type='tree')],
    [FIModelConfig('T-Test', MDI, None, True, model_type='linear')],
]
