import numpy as np
from sklearn.ensemble import RandomForestRegressor
#from imodels import (
#    GreedyTreeClassifier, GreedyTreeRegressor, HSTreeClassifierCV, HSTreeRegressorCV
#)
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi, tree_mdi_OOB, tree_perm_importance, tree_shap, r2f

ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33,'random_state':42})]
]

FI_ESTIMATORS = [
    [FIModelConfig('r2f_aicc_0.33', r2f, model_type='tree',other_params = {'criterion':'aic_c','alpha':0.33})],
    [FIModelConfig('r2f_aicc_0.5', r2f, model_type='tree',other_params = {'criterion':'aic_c','alpha':0.5})],
    [FIModelConfig('r2f_aicc_0.67', r2f, model_type='tree',other_params = {'criterion':'aic_c','alpha':0.67})],
    [FIModelConfig('r2f_aicc_no', r2f, model_type='tree',other_params = {'criterion':'aic_c','alpha':0.5,'split_data':False})],
    [FIModelConfig('r2f_bic_0.33', r2f, model_type='tree',other_params = {'criterion':'bic','alpha':0.33})],
    [FIModelConfig('r2f_bic_0.5', r2f, model_type='tree',other_params = {'criterion':'bic','alpha':0.5})],
    [FIModelConfig('r2f_bic_0.67', r2f, model_type='tree',other_params = {'criterion':'bic','alpha':0.67})],
    [FIModelConfig('r2f_bic_no', r2f, model_type='tree',other_params = {'criterion':'bic','alpha':0.5,'split_data':False})],
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')],
    [FIModelConfig('Permutation', tree_perm_importance, model_type='tree')],
    [FIModelConfig('TreeSHAP', tree_shap, model_type='tree')]
]
