# Note: This is currently just a test file with dummy/placeholder functions

from functools import partial

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from imodels import (
    GreedyTreeClassifier, GreedyTreeRegressor, HSTreeClassifierCV, HSTreeRegressorCV
)
from util import ModelConfig, FIModelConfig

from nonlinear_significance.scripts.methods import lin_reg_t_test, tree_mdi, perm_importance,tree_shap_mean, tree_feature_significance, optimal_tree_feature_significance
#knockpy_swap_integral

ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor,other_params = {'n_estimators':100, 'min_samples_leaf':5, 'max_features':0.33}, model_type='tree')],
    # [ModelConfig('OLS', LinearRegression, model_type='linear')],
]

FI_ESTIMATORS = [
    [FIModelConfig('R2F_max', tree_feature_significance, None, True, model_type='tree',other_params={'max_components': 'max', 'normalize': False})],
    [FIModelConfig('R2F_05', tree_feature_significance, None, True, model_type='tree', other_params={'max_components': 0.5, 'normalize': False})],
    [FIModelConfig('sswR2F_max', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'sequential_stepwise', 'max_components': 'max', 'normalize': False})],
    [FIModelConfig('sswR2F_05', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'sequential_stepwise', 'max_components': 0.5, 'normalize': False})],
    [FIModelConfig('R2F_ridge_pca', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'ridge', 'max_components': 0.5, 'normalize': False})],
    [FIModelConfig('swR2F_05', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'stepwise', 'max_components': 0.5, 'normalize': False})],
    # [FIModelConfig('T-Test', lin_reg_t_test, None, True, model_type='linear')],
    [FIModelConfig('MDI', tree_mdi, None, False, model_type='tree')],
    [FIModelConfig('Permutation', perm_importance, None, False, model_type='tree')],
]
