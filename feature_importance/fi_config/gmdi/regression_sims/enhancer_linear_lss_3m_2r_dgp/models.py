from sklearn.ensemble import RandomForestRegressor
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import GMDI_pipeline, tree_mdi, tree_mdi_OOB, tree_mda, tree_shap


ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33, 'random_state': 27})]
]

FI_ESTIMATORS = [
    [FIModelConfig('GMDI_ridge', GMDI_pipeline, model_type='tree')],
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')],
    [FIModelConfig('MDA', tree_mda, model_type='tree')],
    [FIModelConfig('TreeSHAP', tree_shap, model_type='tree')],
]
