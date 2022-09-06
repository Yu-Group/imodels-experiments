import numpy as np
from sklearn.ensemble import RandomForestRegressor
from imodels import (
    GreedyTreeClassifier, GreedyTreeRegressor, HSTreeClassifierCV, HSTreeRegressorCV
)
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi, tree_mdi_OOB, tree_perm_importance, tree_shap, r2f


ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33}, model_type='tree')]
]

FI_ESTIMATORS = [
    [FIModelConfig('r2f', r2f, model_type='tree')],
    
    [FIModelConfig('r2f (Without Refit)', r2f, model_type='tree', other_params={'refit': False})],
    [FIModelConfig('r2f (Without Xk)', r2f, model_type='tree', other_params={'add_raw': False})],
    
    [FIModelConfig('r2f (AIC)', r2f, model_type='tree', other_params={'criterion': 'aic'})],
    [FIModelConfig('r2f (CV)', r2f, model_type='tree', other_params={'criterion': 'cv', 'alpha': 1})],
]
