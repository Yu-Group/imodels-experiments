import copy
import numpy as np
# from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
# from sklearn.utils.extmath import softmax
from feature_importance.util import ModelConfig, FIModelConfig
from sklearn.ensemble import RandomForestRegressor
from imodels.importance.rf_plus import RandomForestPlusRegressor
from feature_importance.scripts.competing_methods_local import *
      


ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33})],
    [ModelConfig('RF_plus', RandomForestPlusRegressor, model_type='t_plus',
                other_params={'rf_model': RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.33)})],
]

FI_ESTIMATORS = [
    [FIModelConfig('LFI_with_raw_RF', LFI_test_evaluation_RF, model_type='tree', splitting_strategy = "train-test")],
    # [FIModelConfig('LFI_with_raw_CV_RF', LFI_test_evaluation_RF, model_type='tree', splitting_strategy = "train-test", other_params={"cv_ridge": 5, "calc_loo_coef":False})],
    [FIModelConfig('MDI_RF', LFI_test_evaluation_RF, model_type='tree', splitting_strategy = "train-test", other_params={"include_raw": False, "cv_ridge": 0, "calc_loo_coef":False, "sample_split":"inbag"})],
    [FIModelConfig('LFI_with_raw_OOB_RF', LFI_test_evaluation_RF, model_type='tree', splitting_strategy = "train-test", other_params={"sample_split":"oob", "fit_on":"test", "calc_loo_coef":False})],
    [FIModelConfig('TreeSHAP_RF', tree_shap_evaluation_RF, model_type='tree', splitting_strategy = "train-test")],
    [FIModelConfig('LFI_with_raw_RF_plus', LFI_evaluation_RF_plus, model_type='t_plus', splitting_strategy = "train-test")],
    [FIModelConfig('Kernel_SHAP_RF_plus', kernel_shap_evaluation_RF_plus, model_type='t_plus', splitting_strategy = "train-test")],
    [FIModelConfig('LIME_RF_plus', lime_evaluation_RF_plus, model_type='t_plus', splitting_strategy = "train-test")],
]