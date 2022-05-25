import numpy as np
from sklearn.ensemble import RandomForestRegressor
from imodels import (
    GreedyTreeClassifier, GreedyTreeRegressor, HSTreeClassifierCV, HSTreeRegressorCV
)
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi, tree_mdi_OOB, tree_perm_importance, tree_shap, tree_feature_significance


ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33}, model_type='tree')]
]

FI_ESTIMATORS = [
    [FIModelConfig('r2f', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'lasso', 'max_components_type': 'minfracnp', 'fraction_chosen': 0.5, 'criteria': 'bic', 'refit': True})],
    
    [FIModelConfig('r2f (Without Refit)', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'lasso', 'max_components_type': 'minfracnp', 'fraction_chosen': 0.5, 'criteria': 'bic', 'refit': False})],
    [FIModelConfig('r2f (Without Xk)', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'lasso', 'max_components_type': 'minfracnp', 'fraction_chosen': 0.5, 'criteria': 'bic', 'refit': True, 'add_linear': False})],
    
    [FIModelConfig('r2f (AIC)', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'lasso', 'max_components_type': 'minfracnp', 'fraction_chosen': 0.5, 'criteria': 'aic', 'refit': True})],
    [FIModelConfig('r2f (CV)', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'lasso', 'max_components_type': 'minnp', 'fraction_chosen': 1.0, 'criteria': 'cv', 'refit': True})],
]
