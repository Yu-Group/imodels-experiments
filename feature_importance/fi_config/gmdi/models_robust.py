from sklearn.ensemble import RandomForestRegressor
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mda,lin_reg_marginal_t_test,tree_mdi, tree_mdi_OOB, tree_perm_importance, tree_shap,gjMDI
from imodels.importance.r2f_exp_cleaned import GMDI_pipeline, RidgePPM, RobustLOOPPM, huber_loss
from feature_importance.scripts.metrics import corr
import numpy as np
ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33, 'random_state': 27})]
]


FI_ESTIMATORS = [
    [FIModelConfig('GMDI_ridge_loo_r2', GMDI_pipeline, model_type='tree')],
    [FIModelConfig('GMDI_Huber_loo_huber_loss',GMDI_pipeline,model_type = 'tree',other_params = {'partial_prediction_model': RobustLOOPPM(max_iter = 2000), 'scoring_fn': huber_loss})],
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')],
    [FIModelConfig('Permutation', tree_perm_importance, model_type='tree')],
    [FIModelConfig('MDA', tree_mda, model_type='tree')],
    [FIModelConfig('TreeSHAP', tree_shap, model_type='tree')]
]
