# Note: This is currently just a test file with dummy/placeholder functions

from functools import partial

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS
from imodels import (
    GreedyTreeClassifier, GreedyTreeRegressor, HSTreeClassifierCV, HSTreeRegressorCV
)
from util import ModelConfig, FIModelConfig

import sys
sys.path.append("../../")
from nonlinear_significance.scripts.methods import lin_reg_t_test, tree_mdi, perm_importance, knockpy_swap_integral, tree_shap_mean

RANDOM_FOREST_DEFAULT_KWARGS = {'random_state': 0}
ESTIMATORS_CLASSIFICATION = [
    [ModelConfig('CART', GreedyTreeClassifier, 'max_depth', n, model_type='tree')
     for n in [1, 2, 3, 5, 7, 10]],
    # [ModelConfig('OLS', LinearRegression, model_type='linear')],
    # [ModelConfig('HSCART', partial(HSTreeClassifierCV, estimator=DecisionTreeClassifier(max_depth=n)),
    #              'max_depth', n)
    #  for n in [1, 2, 3, 5, 7, 10]],
    # [ModelConfig('Random_Forest', RandomForestClassifier, 'n_estimators', n, other_params=RANDOM_FOREST_DEFAULT_KWARGS)
    #  for n in [3, 10, 25, 50]],
    # [ModelConfig('Gradient_Boosting', GradientBoostingClassifier, 'n_estimators', n,
    #              other_params=RANDOM_FOREST_DEFAULT_KWARGS)
    #  for n in [10, 50, 100]],
]

FI_ESTIMATORS_CLASSIFICATION = [
    # [FIModelConfig('T-Test', log_reg_t_test, None, True, model_type='linear')],
    [FIModelConfig('MDI', tree_mdi, None, False, model_type='tree')],
    [FIModelConfig('Permutation', perm_importance, None, False, model_type='tree')],
    [FIModelConfig('Knockoff', knockpy_swap_integral, None, True, model_type='linear')],
    # [FIModelConfig('TreeSHAP', tree_shap_mean, None, False, model_type='tree')],
]

ENSEMBLE_ESTIMATOR_NUMS = [3, 10, 25, 50]
TREE_DEPTHS = [1, 2, 3, 4, 5, 7, 8, 10, 15, 20, 25]
ESTIMATORS_REGRESSION = [
    [ModelConfig('CART_(MSE)', GreedyTreeRegressor, 'max_depth', n, model_type='tree')
     for n in TREE_DEPTHS],
    [ModelConfig('OLS', LinearRegression, model_type='linear')],
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

FI_ESTIMATORS_REGRESSION = [
    [FIModelConfig('T-Test', lin_reg_t_test, None, True, model_type='linear')],
    [FIModelConfig('MDI', tree_mdi, None, False, model_type='tree')],
    [FIModelConfig('Permutation', perm_importance, None, False, model_type='tree')],
    [FIModelConfig('Knockoff', knockpy_swap_integral, None, True, model_type='linear')],
    # [FIModelConfig('TreeSHAP', tree_shap_mean, None, False, model_type='tree')],
]