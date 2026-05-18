from sklearn.ensemble import RandomForestRegressor
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi_plus_p
from imodels.importance import RidgeRegressorPPM


ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree_deep',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33, 'random_state': 27})],
    [ModelConfig('RF_shallow', RandomForestRegressor, model_type='tree_shallow',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 20, 'max_features': 0.33, 'random_state': 27})]
]

FI_ESTIMATORS = [
    [FIModelConfig('MDI+_F', tree_mdi_plus_p, model_type='tree_deep', other_params={'pval_mode': 'f', 'sample_split': 'oob_only', 'prediction_model': RidgeRegressorPPM(loo=False)})],
    [FIModelConfig('MDI+_permute', tree_mdi_plus_p, model_type='tree_deep', other_params={'pval_mode': 'permute', 'sample_split': 'oob', 'prediction_model': RidgeRegressorPPM(loo=False)})],
    [FIModelConfig('MDI+_F_OLS', tree_mdi_plus_p, model_type='tree_shallow', other_params={'pval_mode': 'f', 'sample_split': 'oob_only', 'prediction_model': RidgeRegressorPPM(loo=False, alpha_grid=1e-6, gcv_mode='eigen')})],
    [FIModelConfig('MDI+_permute_OLS', tree_mdi_plus_p, model_type='tree_shallow', other_params={'pval_mode': 'permute', 'sample_split': 'oob', 'prediction_model': RidgeRegressorPPM(loo=False, alpha_grid=1e-6, gcv_mode='eigen')})],
    [FIModelConfig('MDI+_F_with_all_raw', tree_mdi_plus_p, model_type='tree_deep', other_params={'pval_mode': 'f', 'sample_split': 'oob_only', 'prediction_model': RidgeRegressorPPM(loo=False), 'drop_features': False})],
    [FIModelConfig('MDI+_permute_with_all_raw', tree_mdi_plus_p, model_type='tree_deep', other_params={'pval_mode': 'permute', 'sample_split': 'oob', 'prediction_model': RidgeRegressorPPM(loo=False), 'drop_features': False})],
    [FIModelConfig('MDI+_F_OLS_with_all_raw', tree_mdi_plus_p, model_type='tree_shallow', other_params={'pval_mode': 'f', 'sample_split': 'oob_only', 'prediction_model': RidgeRegressorPPM(loo=False, alpha_grid=1e-6, gcv_mode='eigen'), 'drop_features': False})],
    [FIModelConfig('MDI+_permute_OLS_with_all_raw', tree_mdi_plus_p, model_type='tree_shallow', other_params={'pval_mode': 'permute', 'sample_split': 'oob', 'prediction_model': RidgeRegressorPPM(loo=False, alpha_grid=1e-6, gcv_mode='eigen'), 'drop_features': False})],
]
