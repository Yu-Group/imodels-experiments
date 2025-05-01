import os
import sys
import pandas as pd
import numpy as np
import sklearn.base
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_squared_error, log_loss
from functools import reduce

import shap
import lime
import lime.lime_tabular
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusRegressor, RandomForestPlusClassifier
from sklearn.ensemble import RandomForestRegressor
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import *
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score, mean_squared_error
### MOE imports
from imodels.tree.rf_plus.rf_plus.MOE.rfplus_MOE import SklearnRFPlusRegMOE

def tree_shap_evaluation_RF_retrain(X_train, y_train, X_test, fit=None, mode="absolute"):
    """
    Compute average treeshap value across observations.
    Larger absolute values indicate more important features.
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest (tree-based)
    :return: dataframe of shape: (n_samples, n_features)
    """
    explainer = shap.TreeExplainer(fit)
    local_fi_score_train = explainer.shap_values(X_train, check_additivity=False)
    local_fi_score_test = explainer.shap_values(X_test, check_additivity=False)
    if sklearn.base.is_classifier(fit):
        if mode == "absolute":
            return np.abs(local_fi_score_train[:,:,1]), np.abs(local_fi_score_test[:,:,1])
        else:
            return local_fi_score_train[:,:,1], local_fi_score_test[:,:,1]
    if mode == "absolute":
        return np.abs(local_fi_score_train), np.abs(local_fi_score_test)
    else:
        return local_fi_score_train, local_fi_score_test

def mdi_plus_evaluation_RF_retrain(X_train, y_train, X_test, fit=None, mode="absolute"):
    mdi_plus_scores = fit.get_mdi_plus_scores(X_train, y_train)
    local_fi_score_train = np.tile(mdi_plus_scores["importance"].values, (X_train.shape[0], 1))
    local_fi_score_test = np.tile(mdi_plus_scores["importance"].values, (X_test.shape[0], 1))
    if mode == "absolute":
        return np.abs(local_fi_score_train), np.abs(local_fi_score_test)
    else:
        return local_fi_score_train, local_fi_score_test

def lime_evaluation_RF_retrain(X_train, y_train, X_test, fit=None, mode="absolute"):
    train_result = np.zeros((X_train.shape[0], X_train.shape[1]))
    test_result = np.zeros((X_test.shape[0], X_test.shape[1]))
    if sklearn.base.is_classifier(fit):
        task = "classification"
    else:
        task = "regression"
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, verbose=False, mode=task)
    num_features = X_train.shape[1]
    for i in range(X_train.shape[0]):
        if task == "classification":
            exp = explainer.explain_instance(X_train[i, :], fit.predict_proba, num_features=num_features)
        elif task == "regression":
            exp = explainer.explain_instance(X_train[i, :], fit.predict, num_features=num_features)
        original_feature_importance = exp.as_map()[1]
        sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
        for j in range(num_features):
            train_result[i, j] = sorted_feature_importance[j][1]
    for i in range(X_test.shape[0]):
        if task == "classification":
            exp = explainer.explain_instance(X_test[i, :], fit.predict_proba, num_features=num_features)
        elif task == "regression":
            exp = explainer.explain_instance(X_test[i, :], fit.predict, num_features=num_features)
        original_feature_importance = exp.as_map()[1]
        sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
        for j in range(num_features):
            test_result[i, j] = sorted_feature_importance[j][1]
    if mode == "absolute":
        return np.abs(train_result), np.abs(test_result)
    else:
        return train_result, test_result    

def LFI_evaluation_MDIRFPlus_all_ranking_retrain(X_train, y_train, X_test, fit=None, mode="absolute"):
    assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
    rf_plus_mdi = RFPlusMDI(fit, mode = 'only_k', evaluate_on="all")
    local_fi_score_train = rf_plus_mdi.explain_linear_partial(X=X_train, y=y_train, ranking = True)
    local_fi_score_test = rf_plus_mdi.explain_linear_partial(X=X_test, y=None, ranking = True)
    if mode == "absolute":
        return np.abs(local_fi_score_train), np.abs(local_fi_score_test)
    else:
        return local_fi_score_train, local_fi_score_test

def LFI_evaluation_RFPlus_inbag_retrain(X_train, y_train, X_test, fit=None, mode="absolute"):
    assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
    rf_plus_mdi = RFPlusMDI(fit, mode = 'only_k', evaluate_on="inbag")
    local_fi_score_train = rf_plus_mdi.explain_linear_partial(X=X_train, y=y_train)
    local_fi_score_test = rf_plus_mdi.explain_linear_partial(X=X_test, y=None)
    if mode == "absolute":
        return np.abs(local_fi_score_train), np.abs(local_fi_score_test)
    else:
        return local_fi_score_train, local_fi_score_test
    
def LFI_evaluation_RFPlus_inbag_ranking_retrain(X_train, y_train, X_test, fit=None, mode="absolute"):
    assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
    rf_plus_mdi = RFPlusMDI(fit, mode = 'only_k', evaluate_on="inbag")
    local_fi_score_train = rf_plus_mdi.explain_linear_partial(X=X_train, y=y_train, ranking = True)
    local_fi_score_test = rf_plus_mdi.explain_linear_partial(X=X_test, y=None, ranking = True)
    if mode == "absolute":
        return np.abs(local_fi_score_train), np.abs(local_fi_score_test)
    else:
        return local_fi_score_train, local_fi_score_test
