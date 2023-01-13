from sklearn.ensemble import RandomForestRegressor
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import GMDI_pipeline, tree_mdi, tree_mdi_OOB, tree_mda, tree_shap
from imodels.importance.ppms import RidgePPM

ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 1, 'max_features': 0.33, 'random_state': 27})]
]

FI_ESTIMATORS = [
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('GMDI_basic', GMDI_pipeline, model_type='tree', other_params={'sample_split': 'inbag', 'include_raw': False, 'partial_prediction_model': RidgePPM(loo=False, alpha_grid=1e-6, gcv_mode='eigen')})],
    [FIModelConfig('GMDI_loo', GMDI_pipeline, model_type='tree', other_params={'sample_split': 'loo', 'include_raw': False, 'partial_prediction_model': RidgePPM(loo=False, alpha_grid=1e-6, gcv_mode='eigen')})],
    [FIModelConfig('GMDI_raw', GMDI_pipeline, model_type='tree', other_params={'sample_split': 'inbag', 'include_raw': True, 'partial_prediction_model': RidgePPM(loo=False, alpha_grid=1e-6, gcv_mode='eigen')})],
    [FIModelConfig('GMDI_ridge', GMDI_pipeline, model_type='tree', other_params={'sample_split': 'inbag', 'include_raw': False, 'partial_prediction_model': RidgePPM(loo=False, gcv_mode='eigen')})],
    [FIModelConfig('GMDI_ridge_raw', GMDI_pipeline, model_type='tree', other_params={'sample_split': 'inbag', 'partial_prediction_model': RidgePPM(gcv_mode='eigen')})],
    [FIModelConfig('GMDI_ridge_loo', GMDI_pipeline, model_type='tree', other_params={'include_raw': False, 'partial_prediction_model': RidgePPM(gcv_mode='eigen')})],
    [FIModelConfig('GMDI_ridge_raw_oob', GMDI_pipeline, model_type='tree', other_params={'sample_split': 'oob', 'partial_prediction_model': RidgePPM(loo=False, gcv_mode='eigen')})],
    [FIModelConfig('GMDI_ridge_raw_loo', GMDI_pipeline, model_type='tree', other_params={'partial_prediction_model': RidgePPM(gcv_mode='eigen')})],
    [FIModelConfig('GMDI_ols_raw_loo', GMDI_pipeline, model_type='tree', other_params={'sample_split': 'loo', 'include_raw': True, 'partial_prediction_model': RidgePPM(loo=True, alpha_grid=1e-6, gcv_mode='eigen')})],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')]
]
