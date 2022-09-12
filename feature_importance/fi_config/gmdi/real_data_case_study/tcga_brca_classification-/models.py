from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi, tree_mdi_OOB, tree_perm_importance, tree_shap
from sklearn.metrics import r2_score, log_loss
from imodels.importance.r2f_exp_cleaned import GMDI_pipeline, RidgeLOOPPM


ESTIMATORS = [
    [ModelConfig('RF', RandomForestClassifier, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'random_state': 42})],
    [ModelConfig('RF', RandomForestRegressor, model_type='tree_reg',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'random_state': 42})]
]

FI_ESTIMATORS = [
    # [FIModelConfig('GMDI_logistic_auroc', GMDI_pipeline, model_type='tree', other_params = {'regression': False})],
    # [FIModelConfig('GMDI_logistic_logloss', GMDI_pipeline, model_type='tree', ascending=False, other_params = {'regression': False, 'scoring_fn': log_loss})],
    [FIModelConfig('GMDI_ridge', GMDI_pipeline, model_type='tree', other_params = {'regression': False, 'partial_prediction_model': RidgeLOOPPM(), 'scoring_fn': r2_score})],
    # [FIModelConfig('GPermutation_logistic_auroc', GMDI_pipeline, model_type='tree', other_params = {'regression': False, 'mode': 'keep_rest'})],
    # [FIModelConfig('GPermutation_logistic_logloss', GMDI_pipeline, model_type='tree', ascending=False, other_params = {'regression': False, 'mode': 'keep_rest', 'scoring_fn': log_loss})],
    # [FIModelConfig('GPermutation_ridge', GMDI_pipeline, model_type='tree', other_params = {'regression': False, 'mode': 'keep_rest', 'partial_prediction_model': RidgeLOOPPM(), 'scoring_fn': r2_score})],
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree_reg')],
    [FIModelConfig('Permutation', tree_perm_importance, model_type='tree')],
    [FIModelConfig('TreeSHAP', tree_shap, model_type='tree_reg')]
]
