from sklearn.ensemble import RandomForestClassifier
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import GMDI_pipeline, tree_mdi, tree_mdi_OOB, tree_mda, tree_shap
from imodels.importance.gmdi import _fast_r2_score
from imodels.importance.ppms import RidgePPM
from functools import partial


ESTIMATORS = [
    [ModelConfig('RF', RandomForestClassifier, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'random_state': 42})]
]

FI_ESTIMATORS = [
    [FIModelConfig('GMDI_ridge', GMDI_pipeline, model_type='tree', other_params={'task': 'classification', 'partial_prediction_model': RidgePPM(), 'scoring_fns': partial(_fast_r2_score, multiclass=True)})],
    [FIModelConfig('GMDI_logistic_logloss', GMDI_pipeline, model_type='tree', ascending=False, other_params={'task': 'classification'})],
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('MDA', tree_mda, model_type='tree')],
    [FIModelConfig('TreeSHAP', tree_shap, model_type='tree')]
]
