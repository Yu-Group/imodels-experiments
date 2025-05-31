import copy
import numpy as np
# from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
# from sklearn.utils.extmath import softmax
from feature_importance.util import ModelConfig, FIModelConfig
from sklearn.ensemble import RandomForestClassifier
from feature_importance.scripts.competing_methods_local import *
from sklearn.linear_model import Ridge


ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33, 'random_state': 42})]
]

FI_ESTIMATORS = [
    [FIModelConfig('Treeshap', treeshap_score, model_type='tree', base_model="RF", splitting_strategy = "train-test")],
    [FIModelConfig('LIME', lime_score, model_type='tree', base_model="RF", splitting_strategy = "train-test")],
    [FIModelConfig('lmdi', lmdi_score, model_type='tree', base_model="RFPlus_inbag", splitting_strategy = "train-test")],
    [FIModelConfig('lmdi+', lmdi_plus_score, model_type='tree', base_model="RFPlus", splitting_strategy = "train-test")],
]