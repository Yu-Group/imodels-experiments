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

ENSEMBLE_ESTIMATOR_NUMS = [3, 10, 25, 50]
TREE_DEPTHS = [1, 2, 3, 4, 5, 7, 8, 10, 15, 20, 25]
ESTIMATORS = [
    #[ModelConfig('CART_(MSE)', GreedyTreeRegressor, other_params={'min_samples_leaf': 5}, model_type='tree')],
    [ModelConfig('RF',RandomForestRegressor,other_params = {'n_estimators':100, 'min_samples_leaf':5,'max_features':0.33},model_type = 'tree')],
    [ModelConfig('OLS', LinearRegression, model_type='linear')],



    # [ModelConfig('CART_(MSE)', GreedyTreeRegressor, 'max_depth', n, model_type='tree')
    #  for n in TREE_DEPTHS],
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
    #[FIModelConfig('OptimalTreeSig', optimal_tree_feature_significance, None, True, model_type='tree')],
    [FIModelConfig('R2F+ (sw)', tree_feature_significance, None, True, model_type='tree', other_params={'type': "stepwise"})],
    [FIModelConfig('R2F+', tree_feature_significance, None, True, model_type='tree')],
    [FIModelConfig('R2F', tree_feature_significance, None, True, model_type='tree', other_params={'add_linear': False})],
    [FIModelConfig('T-Test', lin_reg_t_test, None, True, model_type='linear')],
    [FIModelConfig('MDI', tree_mdi, None, False, model_type='tree')],
    [FIModelConfig('Permutation', perm_importance, None, False, model_type='tree')],
]
