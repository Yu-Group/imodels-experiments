from sklearn.ensemble import RandomForestRegressor
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi_plus, tree_mdi, tree_mdi_OOB, tree_mda, tree_shap
from imodels.importance.ppms import RidgeRegressorPPM


ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33, 'random_state': 27})]
]

FI_ESTIMATORS = [
    [FIModelConfig('MDI+', tree_mdi_plus, model_type='tree')],
    [FIModelConfig('MDI+_inbag', tree_mdi_plus, model_type='tree', other_params={"sample_split": "inbag", "prediction_model": RidgeRegressorPPM(loo=False)})],
    [FIModelConfig('MDI+_inbag_noraw', tree_mdi_plus, model_type='tree', other_params={"include_raw": False, "sample_split": "inbag", "prediction_model": RidgeRegressorPPM(loo=False)})],
    [FIModelConfig('MDI+_noraw', tree_mdi_plus, model_type='tree', other_params={"include_raw": False})],
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('MDI_with_splits', tree_mdi, model_type='tree', other_params={"include_num_splits": True})],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')],
    [FIModelConfig('MDA', tree_mda, model_type='tree')],
    [FIModelConfig('TreeSHAP', tree_shap, model_type='tree')],
]
