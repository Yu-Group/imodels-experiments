import numpy as np
from sklearn.ensemble import RandomForestRegressor
from imodels import (
    GreedyTreeClassifier, GreedyTreeRegressor, HSTreeClassifierCV, HSTreeRegressorCV
)
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi, tree_mdi_OOB, tree_perm_importance, tree_shap, tree_feature_significance

# N_ESTIMATORS=[50, 100, 500, 1000]
ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33})],
    # [ModelConfig('RF', RandomForestRegressor, model_type='tree', vary_param="n_estimators", vary_param_val=m,
    #              other_params={'min_samples_leaf': 5, 'max_features': 0.33}) for m in N_ESTIMATORS]
]

FI_ESTIMATORS = [
    [FIModelConfig('r2f', tree_feature_significance, model_type='tree', other_params={'type': 'lasso', 'max_components_type': 'minfracnp', 'fraction_chosen': 0.5, 'criteria': 'bic', 'refit': True})],
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')],
    [FIModelConfig('Permutation', tree_perm_importance, model_type='tree')],
    [FIModelConfig('TreeSHAP', tree_shap, model_type='tree')]
]

# FRACTION_CHOSEN=[0.2, 0.5, 0.8]
# FI_ESTIMATORS = [
#     [FIModelConfig('r2f', tree_feature_significance, model_type='tree', vary_param="fraction_chosen", vary_param_val=m, other_params={'type': 'lasso', 'max_components_type': 'minfracnp', 'criteria': 'bic', 'refit': True}) for m in FRACTION_CHOSEN]
# ]
