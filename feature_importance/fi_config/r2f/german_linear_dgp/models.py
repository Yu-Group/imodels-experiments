import numpy as np
from sklearn.ensemble import RandomForestRegressor
from imodels import (
    GreedyTreeClassifier, GreedyTreeRegressor, HSTreeClassifierCV, HSTreeRegressorCV
)
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi, tree_mdi_OOB, tree_perm_importance, tree_shap, tree_feature_significance

ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33})]
]

FI_ESTIMATORS = [
    [FIModelConfig('r2f', tree_feature_significance, None, model_type='tree', other_params={'type': 'lasso', 'max_components_type': 'minfracnp', 'fraction_chosen': 0.5, 'criteria': 'bic', 'refit': True})],
    [FIModelConfig('MDI', tree_mdi, None, model_type='tree')],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, None, model_type='tree')],
    [FIModelConfig('Permutation', tree_perm_importance, None, model_type='tree')],
    [FIModelConfig('TreeSHAP', tree_shap, None, model_type='tree')]
]
