import copy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from sklearn.utils.extmath import softmax
from feature_importance.util import ModelConfig, FIModelConfig
from imodels.importance.rf_plus import RandomForestPlusClassifier
# from imodels.importance.ppms import RidgeClassifierPPM, LogisticClassifierPPMg
from feature_importance.scripts.competing_methods_local import *
      


ESTIMATORS = [
    [ModelConfig('RF', RandomForestClassifier, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'random_state': 42})]
]


FI_ESTIMATORS = [
    [FIModelConfig('LFI_with_raw', LFI_ablation_test_evaluation, model_type='tree', splitting_strategy = "train-test")],
    [FIModelConfig('LFI_without_raw', LFI_ablation_test_evaluation, model_type='tree', splitting_strategy = "train-test", other_params={"include_raw": False})],
    [FIModelConfig('TreeSHAP', tree_shap_local, model_type='tree', splitting_strategy = "train-test")],
    [FIModelConfig('LIME', lime_local, model_type='tree', splitting_strategy = "train-test")],
]
