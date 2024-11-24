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
    [FIModelConfig('Random', random_retrain, model_type='tree', base_model="None", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_inbag_RFPlus', LFI_evaluation_RFPlus_inbag_retrain, model_type='tree', base_model="RFPlus_inbag", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_oob_RFPlus', LFI_evaluation_RFPlus_oob_retrain, model_type='tree', base_model="RFPlus_oob", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_all_RFPlus', LFI_evaluation_RFPlus_all_retrain, model_type='tree', base_model="RFPlus_default", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_inbag_average_RFPlus', LFI_evaluation_RFPlus_inbag_average_retrain, model_type='tree', base_model="RFPlus_inbag", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_oob_average_RFPlus', LFI_evaluation_RFPlus_oob_average_retrain, model_type='tree', base_model="RFPlus_oob", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_all_average_RFPlus', LFI_evaluation_RFPlus_all_average_retrain, model_type='tree', base_model="RFPlus_default", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_inbag_ranking_RFPlus', LFI_evaluation_RFPlus_inbag_ranking_retrain, model_type='tree', base_model="RFPlus_inbag", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_oob_ranking_RFPlus', LFI_evaluation_RFPlus_oob_ranking_retrain, model_type='tree', base_model="RFPlus_oob", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_all_ranking_RFPlus', LFI_evaluation_RFPlus_all_ranking_retrain, model_type='tree', base_model="RFPlus_default", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_inbag_l2_norm_RFPlus', LFI_evaluation_RFPlus_inbag_l2_norm_retrain, model_type='tree', base_model="RFPlus_inbag", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_oob_l2_norm_RFPlus', LFI_evaluation_RFPlus_oob_l2_norm_retrain, model_type='tree', base_model="RFPlus_oob", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_all_l2_norm_RFPlus', LFI_evaluation_RFPlus_all_l2_norm_retrain, model_type='tree', base_model="RFPlus_default", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_inbag_l2_norm_average_RFPlus', LFI_evaluation_RFPlus_inbag_l2_norm_average_retrain, model_type='tree', base_model="RFPlus_inbag", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_oob_l2_norm_average_RFPlus', LFI_evaluation_RFPlus_oob_l2_norm_average_retrain, model_type='tree', base_model="RFPlus_oob", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_all_l2_norm_average_RFPlus', LFI_evaluation_RFPlus_all_l2_norm_average_retrain, model_type='tree', base_model="RFPlus_default", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_inbag_l2_norm_ranking_RFPlus', LFI_evaluation_RFPlus_inbag_l2_norm_ranking_retrain, model_type='tree', base_model="RFPlus_inbag", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_oob_l2_norm_ranking_RFPlus', LFI_evaluation_RFPlus_oob_l2_norm_ranking_retrain, model_type='tree', base_model="RFPlus_oob", splitting_strategy = "train-test")],
    [FIModelConfig('Local_MDI+_fit_on_all_l2_norm_ranking_RFPlus', LFI_evaluation_RFPlus_all_l2_norm_ranking_retrain, model_type='tree', base_model="RFPlus_default", splitting_strategy = "train-test")],
   
]
