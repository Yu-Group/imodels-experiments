from sklearn.ensemble import RandomForestClassifier
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi_plus, tree_mdi, tree_mdi_OOB, tree_mda, tree_shap
from imodels.importance.rf_plus import _fast_r2_score
from imodels.importance.ppms import RidgeClassifierPPM, LogisticClassifierPPM


ESTIMATORS = [
    [ModelConfig('RF', RandomForestClassifier, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'random_state': 27})]
]

FI_ESTIMATORS = [
    [FIModelConfig('MDI+_ridge', tree_mdi_plus, model_type='tree', other_params={'prediction_model': RidgeClassifierPPM(gcv_mode='eigen'), 'scoring_fns': _fast_r2_score})],
    [FIModelConfig('MDI+_ridge_inbag', tree_mdi_plus, model_type='tree', other_params={'prediction_model': RidgeClassifierPPM(loo=False, gcv_mode='eigen'), 'scoring_fns': _fast_r2_score, "sample_split": "inbag"})],
    [FIModelConfig('MDI+_logistic_logloss', tree_mdi_plus, model_type='tree')],
    [FIModelConfig('MDI+_logistic_logloss_inbag', tree_mdi_plus, model_type='tree', other_params={"sample_split": "inbag", "prediction_model": LogisticClassifierPPM(loo=False)})],
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('MDI_with_splits', tree_mdi, model_type='tree', other_params={"include_num_splits": True})],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')],
    [FIModelConfig('MDA', tree_mda, model_type='tree')],
    [FIModelConfig('TreeSHAP', tree_shap, model_type='tree')]
]

