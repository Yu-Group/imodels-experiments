import copy
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi_plus, tree_mdi, tree_mdi_OOB, tree_mda, tree_shap
from sksurv.ensemble import RandomSurvivalForest

from imodels.importance.rf_plus import RandomForestPlusSurvival
from imodels.importance.ppms import CoxnetSurvivalPPM



rf_model = RandomSurvivalForest(n_estimators=100, min_samples_leaf=5, max_features=0.33, random_state=42)
ppm_model = CoxnetSurvivalPPM()
ESTIMATORS = [
    # [ModelConfig('RSF', RandomSurvivalForest, model_type='rsf',
    #              other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33, 'random_state': 27})],
    [ModelConfig('RSF+', RandomForestPlusSurvival, model_type='rsf+',
                 other_params={'rf_model': copy.deepcopy(rf_model),
                               'prediction_model': copy.deepcopy(ppm_model)})],
]

FI_ESTIMATORS = [
    [FIModelConfig('MDI+', tree_mdi_plus, model_type='rsf+', splitting_strategy='train-test',
                   other_params={'refit': False, 'sample_split': None,
                                 'mdiplus_kwargs': {'sample_split': None}})],
    # [FIModelConfig('MDI+_ridge_loo_mae', tree_mdi_plus, model_type='tree', ascending=False, other_params={'scoring_fns': mean_absolute_error})],
    # [FIModelConfig('MDI+_Huber_loo_huber_loss', tree_mdi_plus, model_type='tree', ascending=False, other_params={'prediction_model': RobustRegressorPPM(), 'scoring_fns': huber_loss})],
    # [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    # [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')],
    # [FIModelConfig('MDA', tree_mda, model_type='tree')],
    # [FIModelConfig('TreeSHAP', tree_shap, model_type='tree')]
]
