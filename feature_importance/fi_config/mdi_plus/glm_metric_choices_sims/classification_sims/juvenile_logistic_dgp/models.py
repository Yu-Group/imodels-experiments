from sklearn.ensemble import RandomForestClassifier
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi_plus, tree_mdi, tree_mdi_OOB, tree_mda, tree_shap, tree_mdi_plus_ensemble
from imodels.importance.rf_plus import _fast_r2_score, _neg_log_loss
from imodels.importance.ppms import RidgeClassifierPPM, LogisticClassifierPPM
from sklearn.metrics import roc_auc_score


ESTIMATORS = [
    [ModelConfig('RF', RandomForestClassifier, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'random_state': 27})]
]

FI_ESTIMATORS = [
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('MDI+_ridge_r2', tree_mdi_plus, model_type='tree', other_params={'prediction_model': RidgeClassifierPPM(), 'scoring_fns': _fast_r2_score, 'return_stability_scores': True})],
    [FIModelConfig('MDI+_logistic_ridge_logloss', tree_mdi_plus, model_type='tree', other_params={'return_stability_scores': True})],
    [FIModelConfig('MDI+_logistic_ridge_auroc', tree_mdi_plus, model_type='tree', other_params={'scoring_fns': roc_auc_score, 'return_stability_scores': True})],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')]
]
