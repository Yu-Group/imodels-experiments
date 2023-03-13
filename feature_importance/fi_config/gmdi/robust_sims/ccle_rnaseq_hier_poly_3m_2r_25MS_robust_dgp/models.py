from sklearn.ensemble import RandomForestRegressor
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_gmdi, tree_mdi, tree_mdi_OOB, tree_mda, tree_shap

from imodels.importance.ppms import RobustRegressorPPM, LassoRegressorPPM, huber_loss
from sklearn.metrics import mean_absolute_error


ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33, 'random_state': 27})]
]

FI_ESTIMATORS = [
    [FIModelConfig('GMDI_ridge_loo_r2', tree_gmdi, model_type='tree')],
    [FIModelConfig('GMDI_ridge_loo_mae', tree_gmdi, model_type='tree', ascending=False, other_params={'scoring_fns': mean_absolute_error})],
    [FIModelConfig('GMDI_Huber_loo_huber_loss', tree_gmdi, model_type='tree', ascending=False, other_params={'prediction_model': RobustRegressorPPM(), 'scoring_fns': huber_loss})],
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')],
    [FIModelConfig('MDA', tree_mda, model_type='tree')],
    [FIModelConfig('TreeSHAP', tree_shap, model_type='tree')]
]
