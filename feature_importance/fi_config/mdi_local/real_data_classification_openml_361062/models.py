import copy
import numpy as np
# from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
# from sklearn.utils.extmath import softmax
from feature_importance.util import ModelConfig, FIModelConfig
from sklearn.ensemble import RandomForestClassifier
from feature_importance.scripts.competing_methods_local import *
from sklearn.linear_model import Ridge

ESTIMATORS = [
    [ModelConfig('RF', RandomForestClassifier, model_type='tree',
                other_params={'n_estimators': 100, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'random_state': 42})],
]


FI_ESTIMATORS = [
    [FIModelConfig('TreeSHAP_RF', tree_shap_evaluation_RF_retrain, model_type='tree', base_model="RF", splitting_strategy = "train-test")],
    [FIModelConfig('LIME_RF', lime_evaluation_RF_retrain, model_type='tree', base_model="RF", splitting_strategy = "train-test")],
    [FIModelConfig('MDI', LFI_evaluation_RFPlus_inbag_retrain, model_type='tree', base_model="RFPlus_inbag", splitting_strategy = "train-test")],
    [FIModelConfig('MDI+', mdi_plus_evaluation_RF_retrain, model_type='tree', base_model="RFPlus_mdi_plus", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_MDI_elasticnet_fit_on_all_ranking_RFPlus', LFI_evaluation_MDIRFPlus_all_ranking_retrain, model_type='tree', base_model="RFPlus_elastic", splitting_strategy = "train-test")],
    [FIModelConfig('Ablation_model1', LFI_evaluation_MDIRFPlus_all_ranking_retrain, model_type='tree', base_model="Ablation_model1", splitting_strategy = "train-test")],
    [FIModelConfig('Ablation_model2', LFI_evaluation_MDIRFPlus_all_ranking_retrain, model_type='tree', base_model="Ablation_model2", splitting_strategy = "train-test")],
]