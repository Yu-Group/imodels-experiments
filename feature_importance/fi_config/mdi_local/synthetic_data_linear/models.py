from sklearn.ensemble import RandomForestRegressor
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods_local import *

ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33, 'random_state': 42})]
]

FI_ESTIMATORS = [
    [FIModelConfig('TreeSHAP_RF', tree_shap_evaluation_RF, model_type='tree', base_model="RF", splitting_strategy = "test-300")],
    [FIModelConfig('Local_MDI+_fit_on_OOB_RFPlus_l2_norm', LFI_evaluation_RFPlus_oob_l2_norm_sign, model_type='tree', base_model="RFPlus_oob", splitting_strategy = "test-300")],
    [FIModelConfig('Local_MDI+_fit_on_all_evaluate_on_all_RFPlus_l2_norm', LFI_evaluation_RFPlus_all_l2_norm_sign, model_type='tree', base_model="RFPlus_default", splitting_strategy = "test-300")],
    [FIModelConfig('Local_MDI+_fit_on_all_evaluate_on_oob_RFPlus_l2_norm', LFI_evaluation_RFPlus_oob_l2_norm_sign, model_type='tree', base_model="RFPlus_default", splitting_strategy = "test-300")],
    [FIModelConfig('Local_MDI+_fit_on_OOB_RFPlus', LFI_evaluation_RFPlus_oob, model_type='tree', base_model="RFPlus_oob", splitting_strategy = "test-300")],
    [FIModelConfig('Local_MDI+_fit_on_all_evaluate_on_all_RFPlus', LFI_evaluation_RFPlus_all, model_type='tree', base_model="RFPlus_default", splitting_strategy = "test-300")],
    [FIModelConfig('Local_MDI+_fit_on_all_evaluate_on_oob_RFPlus', LFI_evaluation_RFPlus_oob, model_type='tree', base_model="RFPlus_default", splitting_strategy = "test-300")],
    [FIModelConfig('Kernel_SHAP_RF_plus', kernel_shap_evaluation_RF_plus, model_type='tree', base_model="RFPlus_default", splitting_strategy = "test-300")],
    [FIModelConfig('LIME_RF_plus', lime_evaluation_RF_plus, model_type='tree', base_model="RFPlus_default", splitting_strategy = "test-300")],
    [FIModelConfig('Random', random, model_type='tree', base_model="None", splitting_strategy = "test-300")],
]