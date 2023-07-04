from sklearn.ensemble import RandomForestClassifier
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi_plus, tree_mdi, tree_mdi_OOB, tree_mda, tree_shap
from imodels.importance.rf_plus import _fast_r2_score
from imodels.importance.ppms import RidgeClassifierPPM


ESTIMATORS = [
    [ModelConfig('RF', RandomForestClassifier, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'random_state': 27})]
]

FI_ESTIMATORS = [
    [FIModelConfig('MDI+_ridge', tree_mdi_plus, model_type='tree', other_params={'prediction_model': RidgeClassifierPPM(), 'scoring_fns': _fast_r2_score})],
    [FIModelConfig('MDI+_logistic_logloss', tree_mdi_plus, model_type='tree')],
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')],
    [FIModelConfig('MDA', tree_mda, model_type='tree')],
    [FIModelConfig('TreeSHAP', tree_shap, model_type='tree')]
]
