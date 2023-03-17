from sklearn.ensemble import RandomForestRegressor
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi_plus, tree_mdi, tree_mdi_OOB, tree_mda, tree_shap
from imodels.importance.ppms import RidgeRegressorPPM

ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33, 'random_state': 27})]
]

FI_ESTIMATORS = [
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('MDI+_loo', tree_mdi_plus, model_type='tree', other_params={'sample_split': 'loo', 'include_raw': False, 'prediction_model': RidgeRegressorPPM(loo=True, alpha_grid=1e-6, gcv_mode='eigen')})],
    [FIModelConfig('MDI+_raw', tree_mdi_plus, model_type='tree', other_params={'sample_split': 'inbag', 'include_raw': True, 'prediction_model': RidgeRegressorPPM(loo=False, alpha_grid=1e-6, gcv_mode='eigen')})],
    [FIModelConfig('MDI+_ridge_raw_loo', tree_mdi_plus, model_type='tree', other_params={'prediction_model': RidgeRegressorPPM(gcv_mode='eigen')})],
    [FIModelConfig('MDI+_ols_raw_loo', tree_mdi_plus, model_type='tree', other_params={'sample_split': 'loo', 'include_raw': True, 'prediction_model': RidgeRegressorPPM(loo=True, alpha_grid=1e-6, gcv_mode='eigen')})],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')]
]
