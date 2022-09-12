from sklearn.ensemble import RandomForestRegressor
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi, tree_mdi_OOB, tree_perm_importance, tree_shap
from imodels.importance.r2f_exp_cleaned import GMDI_pipeline


ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33, 'random_state': 42})]
]

FI_ESTIMATORS = [
    [FIModelConfig('GMDI_ridge', GMDI_pipeline, model_type='tree')],
    # [FIModelConfig('GPermutation_ridge', GMDI_pipeline, model_type='tree', other_params = {'mode': 'keep_rest'})],
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    # [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')],
    # [FIModelConfig('Permutation', tree_perm_importance, model_type='tree')],
    # [FIModelConfig('TreeSHAP', tree_shap, model_type='tree')]
]