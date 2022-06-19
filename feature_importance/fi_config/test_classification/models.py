import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from imodels import (
    GreedyTreeClassifier, GreedyTreeRegressor, HSTreeClassifierCV, HSTreeRegressorCV
)
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi, tree_mdi_OOB, tree_perm_importance, tree_shap, r2f, gMDI, gjMDI

# N_ESTIMATORS=[50, 100, 500, 1000]
ESTIMATORS = [
    [ModelConfig('RF', RandomForestClassifier, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 1, 'max_features': 'sqrt'})],
]

FI_ESTIMATORS = [
    [FIModelConfig('gjMDI_ridge', gjMDI, model_type='tree', other_params={'scoring_type': 'ridge', 'criterion': 'gcv', 'normalize_raw': True})],
    [FIModelConfig('gjMDI_log', gjMDI, model_type='tree', other_params={'scoring_type': 'logistic', 'criterion': 'gcv', 'normalize_raw': True})],
    # [FIModelConfig('r2f', r2f, model_type='tree', other_params={'criterion': 'cv_1se'})],
    # [FIModelConfig('r2f_ridge_1se', r2f, model_type='tree', other_params={'scoring_type': 'ridge', 'criterion': 'gcv_1se'})],
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')],
    [FIModelConfig('Permutation', tree_perm_importance, model_type='tree')],
    [FIModelConfig('TreeSHAP', tree_shap, model_type='tree')]
]
