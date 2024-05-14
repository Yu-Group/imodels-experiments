import copy
import numpy as np
from feature_importance.util import ModelConfig, FIModelConfig
from sklearn.ensemble import RandomForestRegressor
from feature_importance.scripts.competing_methods_local import *
from sklearn.linear_model import Ridge


ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33, 'random_state': 42})]
]

FI_ESTIMATORS = [
    [FIModelConfig('TreeSHAP_RF', tree_shap_evaluation_RF, model_type='tree', splitting_strategy = "train-test")],
    [FIModelConfig('LFI_fit_on_inbag_RF', LFI_evaluation_RF_MDI, model_type='tree', splitting_strategy = "train-test", ascending = False, other_params={"include_raw":False, "fit_on":"inbag", "prediction_model": Ridge(alpha=1e-6)})],
    [FIModelConfig('LFI_fit_on_OOB_RF', LFI_evaluation_RF_OOB, model_type='tree', splitting_strategy = "train-test",  ascending = False, other_params={"fit_on":"oob"})],
    [FIModelConfig('LFI_evaluate_on_all_RF_plus', LFI_evaluation_RF_plus, model_type='tree', splitting_strategy = "train-test", ascending = False)],
    [FIModelConfig('LFI_evaluate_on_oob_RF_plus', LFI_evaluation_RF_plus_OOB, model_type='tree', splitting_strategy = "train-test", ascending = False)],
    [FIModelConfig('Kernel_SHAP_RF_plus', kernel_shap_evaluation_RF_plus, model_type='tree', splitting_strategy = "train-test")],
    [FIModelConfig('LIME_RF_plus', lime_evaluation_RF_plus, model_type='tree', splitting_strategy = "train-test")],
]