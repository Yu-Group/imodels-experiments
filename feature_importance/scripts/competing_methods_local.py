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
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusRegressor, RandomForestPlusClassifier
from sklearn.ensemble import RandomForestRegressor
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import *

def treeshap_score(X_train, y_train, X_test, model=None, absolute=True):
    explainer = shap.TreeExplainer(model)
    lfi_train = explainer.shap_values(X_train, check_additivity=False)
    lfi_test = explainer.shap_values(X_test, check_additivity=False)
    if sklearn.base.is_classifier(model):
        lfi_train = lfi_train[:,:,1]
        lfi_test = lfi_test[:,:,1]
    if absolute:
        return np.abs(lfi_train), np.abs(lfi_test)
    else:
        return lfi_train, lfi_test


def lime_score(X_train, y_train, X_test, model=None, absolute=True):
    lfi_train = np.zeros((X_train.shape[0], X_train.shape[1]))
    lfi_test = np.zeros((X_test.shape[0], X_test.shape[1]))
    if sklearn.base.is_classifier(model):
        task = "classification"
    else:
        task = "regression"
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, verbose=False, mode=task)
    num_features = X_train.shape[1]

    for i in range(X_train.shape[0]):
        if task == "classification":
            exp = explainer.explain_instance(X_train[i, :], model.predict_proba, num_features=num_features)
        elif task == "regression":
            exp = explainer.explain_instance(X_train[i, :], model.predict, num_features=num_features)
        original_feature_importance = exp.as_map()[1]
        sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
        for j in range(num_features):
            lfi_train[i, j] = sorted_feature_importance[j][1]

    for i in range(X_test.shape[0]):
        if task == "classification":
            exp = explainer.explain_instance(X_test[i, :], model.predict_proba, num_features=num_features)
        elif task == "regression":
            exp = explainer.explain_instance(X_test[i, :], model.predict, num_features=num_features)
        original_feature_importance = exp.as_map()[1]
        sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
        for j in range(num_features):
            lfi_test[i, j] = sorted_feature_importance[j][1]
    if absolute:
        return np.abs(lfi_train), np.abs(lfi_test)
    else:
        return lfi_train, lfi_test


def lmdi_score(X_train, y_train, X_test, model=None, absolute=True):
    assert isinstance(model, RandomForestPlusRegressor) or isinstance(model, RandomForestPlusClassifier)
    rf_plus_mdi = LMDIPlus(model, evaluate_on="inbag")
    lfi_train = rf_plus_mdi.get_lmdi_plus_scores(X=X_train, y=y_train)
    lfi_test = rf_plus_mdi.get_lmdi_plus_scores(X=X_test, y=None)
    if absolute:
        return np.abs(lfi_train), np.abs(lfi_test)
    else:
        return lfi_train, lfi_test

def lmdi_plus_score(X_train, y_train, X_test, model=None, absolute=True):
    assert isinstance(model, RandomForestPlusRegressor) or isinstance(model, RandomForestPlusClassifier)
    rf_plus_mdi = LMDIPlus(model, evaluate_on="all")
    lfi_train = rf_plus_mdi.get_lmdi_plus_scores(X=X_train, y=y_train, ranking = True)
    lfi_test = rf_plus_mdi.get_lmdi_plus_scores(X=X_test, y=None, ranking = True)
    if absolute:
        return np.abs(lfi_train), np.abs(lfi_test)
    else:
        return lfi_train, lfi_test

# def mdi_plus_evaluation_RF_retrain(X_train, y_train, X_test, fit=None, mode="absolute"):
#     mdi_plus_scores = fit.get_mdi_plus_scores(X_train, y_train)
#     local_fi_score_train = np.tile(mdi_plus_scores["importance"].values, (X_train.shape[0], 1))
#     local_fi_score_test = np.tile(mdi_plus_scores["importance"].values, (X_test.shape[0], 1))
#     if mode == "absolute":
#         return np.abs(local_fi_score_train), np.abs(local_fi_score_test)
#     else:
#         return local_fi_score_train, local_fi_score_test

# def lime_evaluation_RF_retrain(X_train, y_train, X_test, fit=None, mode="absolute"):
#     train_result = np.zeros((X_train.shape[0], X_train.shape[1]))
#     test_result = np.zeros((X_test.shape[0], X_test.shape[1]))
#     if sklearn.base.is_classifier(fit):
#         task = "classification"
#     else:
#         task = "regression"
#     explainer = lime.lime_tabular.LimeTabularExplainer(X_train, verbose=False, mode=task)
#     num_features = X_train.shape[1]
#     for i in range(X_train.shape[0]):
#         if task == "classification":
#             exp = explainer.explain_instance(X_train[i, :], fit.predict_proba, num_features=num_features)
#         elif task == "regression":
#             exp = explainer.explain_instance(X_train[i, :], fit.predict, num_features=num_features)
#         original_feature_importance = exp.as_map()[1]
#         sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
#         for j in range(num_features):
#             train_result[i, j] = sorted_feature_importance[j][1]
#     for i in range(X_test.shape[0]):
#         if task == "classification":
#             exp = explainer.explain_instance(X_test[i, :], fit.predict_proba, num_features=num_features)
#         elif task == "regression":
#             exp = explainer.explain_instance(X_test[i, :], fit.predict, num_features=num_features)
#         original_feature_importance = exp.as_map()[1]
#         sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
#         for j in range(num_features):
#             test_result[i, j] = sorted_feature_importance[j][1]
#     if mode == "absolute":
#         return np.abs(train_result), np.abs(test_result)
#     else:
#         return train_result, test_result    

# def LFI_evaluation_MDIRFPlus_all_ranking_retrain(X_train, y_train, X_test, fit=None, mode="absolute"):
#     assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
#     rf_plus_mdi = RFPlusMDI(fit, mode = 'only_k', evaluate_on="all")
#     local_fi_score_train = rf_plus_mdi.explain_linear_partial(X=X_train, y=y_train, ranking = True)
#     local_fi_score_test = rf_plus_mdi.explain_linear_partial(X=X_test, y=None, ranking = True)
#     if mode == "absolute":
#         return np.abs(local_fi_score_train), np.abs(local_fi_score_test)
#     else:
#         return local_fi_score_train, local_fi_score_test

# def LFI_evaluation_RFPlus_inbag_retrain(X_train, y_train, X_test, fit=None, mode="absolute"):
#     assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
#     rf_plus_mdi = RFPlusMDI(fit, mode = 'only_k', evaluate_on="inbag")
#     local_fi_score_train = rf_plus_mdi.explain_linear_partial(X=X_train, y=y_train)
#     local_fi_score_test = rf_plus_mdi.explain_linear_partial(X=X_test, y=None)
#     if mode == "absolute":
#         return np.abs(local_fi_score_train), np.abs(local_fi_score_test)
#     else:
#         return local_fi_score_train, local_fi_score_test