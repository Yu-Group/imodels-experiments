from sklearn.ensemble import RandomForestClassifier
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi_plus_p
from imodels.importance import LogisticClassifierPPM


ESTIMATORS = [
    [ModelConfig('RF', RandomForestClassifier, model_type='tree_deep',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'random_state': 27})],
    [ModelConfig('RF_shallow', RandomForestClassifier, model_type='tree_shallow',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 20, 'max_features': 'sqrt', 'random_state': 27})]
]

FI_ESTIMATORS = [
    [FIModelConfig('MDI+_chi2', tree_mdi_plus_p, model_type='tree_deep', other_params={'pval_mode': 'f', 'sample_split': 'oob_only', 'prediction_model': LogisticClassifierPPM(loo=False)})],
    [FIModelConfig('MDI+_permute', tree_mdi_plus_p, model_type='tree_deep', other_params={'pval_mode': 'permute', 'sample_split': 'oob', 'prediction_model': LogisticClassifierPPM(loo=False)})],
    [FIModelConfig('MDI+_chi2_OLS', tree_mdi_plus_p, model_type='tree_shallow', other_params={'pval_mode': 'f', 'sample_split': 'oob_only', 'prediction_model': LogisticClassifierPPM(loo=False, alpha_grid=1e-6)})],
    [FIModelConfig('MDI+_permute_OLS', tree_mdi_plus_p, model_type='tree_shallow', other_params={'pval_mode': 'permute', 'sample_split': 'oob', 'prediction_model': LogisticClassifierPPM(loo=False, alpha_grid=1e-6)})],
    [FIModelConfig('MDI+_chi2_with_all_raw', tree_mdi_plus_p, model_type='tree_deep', other_params={'pval_mode': 'f', 'sample_split': 'oob_only', 'prediction_model': LogisticClassifierPPM(loo=False), 'drop_features': False})],
    [FIModelConfig('MDI+_permute_with_all_raw', tree_mdi_plus_p, model_type='tree_deep', other_params={'pval_mode': 'permute', 'sample_split': 'oob', 'prediction_model': LogisticClassifierPPM(loo=False), 'drop_features': False})],
    [FIModelConfig('MDI+_chi2_OLS_with_all_raw', tree_mdi_plus_p, model_type='tree_shallow', other_params={'pval_mode': 'f', 'sample_split': 'oob_only', 'prediction_model': LogisticClassifierPPM(loo=False, alpha_grid=1e-6), 'drop_features': False})],
    [FIModelConfig('MDI+_permute_OLS_with_all_raw', tree_mdi_plus_p, model_type='tree_shallow', other_params={'pval_mode': 'permute', 'sample_split': 'oob', 'prediction_model': LogisticClassifierPPM(loo=False, alpha_grid=1e-6), 'drop_features': False})],
]
