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
    [FIModelConfig('R2F_max', tree_feature_significance, None, True, model_type='tree',other_params={'max_components': 'max'})],
    [FIModelConfig('R2F_05', tree_feature_significance, None, True, model_type='tree', other_params={'max_components': 0.5})],
    [FIModelConfig('sswR2F_max', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'sequential_stepwise', 'max_components': 'max'})],
    [FIModelConfig('sswR2F_05', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'sequential_stepwise', 'max_components': 0.5})],
    [FIModelConfig('R2F_ridge_pca', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'ridge', 'max_components': 0.5})],
    [FIModelConfig('swR2F_05', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'stepwise', 'max_components': 0.5})],
    [FIModelConfig('T-Test', lin_reg_t_test, None, True, model_type='linear')],
    [FIModelConfig('MDI', tree_mdi, None, False, model_type='tree')],
    [FIModelConfig('Permutation', perm_importance, None, False, model_type='tree')],
]

# FI_ESTIMATORS = [
#     # [FIModelConfig('OptimalTreeSig', optimal_tree_feature_significance, None, True, model_type='tree')],
#     [FIModelConfig('TreeSig', tree_feature_significance, None, True, model_type='tree')],
#     [FIModelConfig('T-Test', lin_reg_t_test, None, True, model_type='linear')],
#     [FIModelConfig('MDI', tree_mdi, None, False, model_type='tree')],
#     [FIModelConfig('Permutation', perm_importance, None, False, model_type='tree')],
#     [FIModelConfig('TreeSHAP', tree_shap_mean, None, False, model_type='tree')],
#     [FIModelConfig('Boruta', boruta_rank, None, False, model_type='rf')],
#     [FIModelConfig('FOCI', foci_rank, None, False, model_type='linear')],  # model_type=None in reality
#     [FIModelConfig('Knockoff', knockpy_swap_integral, None, True, model_type='tree', other_params={'knockoff_fdr':0.05})]
# ]

# MAX_COMPONENTS=[0.1, 0.2, 0.3, 0.4, 0.5]
# FI_ESTIMATORS = [
#     # [FIModelConfig('OptimalTreeSig', optimal_tree_feature_significance, None, True, model_type='tree', vary_param="max_components", vary_param_val=m) for m in MAX_COMPONENTS],
#     [FIModelConfig('TreeSig', tree_feature_significance, None, True, model_type='tree', vary_param="max_components", vary_param_val=m) for m in MAX_COMPONENTS],
#     # [FIModelConfig('T-Test', lin_reg_t_test, None, True, model_type='linear')],
#     # [FIModelConfig('MDI', tree_mdi, None, False, model_type='tree')],
#     # [FIModelConfig('Permutation', perm_importance, None, False, model_type='tree')],
# ]
