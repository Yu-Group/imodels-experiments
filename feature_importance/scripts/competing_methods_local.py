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

### Helper function that mask the matrix
def feature_importance_mask(feature_importance, mask_matrix, mode, mask_to = "zero"):
    assert mode in ["positive", "negative"]
    assert mask_to in ["zero", "inf"]
    masked_feature_importance = feature_importance.copy()
    if mode == "positive":
        mask = mask_matrix > 0
    elif mode == "negative":
        mask = mask_matrix < 0
    if mask_to == "zero":
        masked_feature_importance[~mask] = 0
    else:
        masked_feature_importance[~mask] = sys.maxsize - 1
    return masked_feature_importance

#### Baseline Methods
def random(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset, fit=None, mode="absolute"):
    local_fi_score_train = None
    local_fi_score_train_subset = np.random.randn(*X_train_subset.shape)
    local_fi_score_test = np.random.randn(*X_test.shape)
    local_fi_score_test_subset = np.random.randn(*X_test_subset.shape)
    if mode == "absolute":
        return None, np.abs(local_fi_score_train_subset), np.abs(local_fi_score_test), np.abs(local_fi_score_test_subset)
    else:
        local_fi_score_train_subset = feature_importance_mask(local_fi_score_train_subset, local_fi_score_train_subset, mode, mask_to = "zero")
        local_fi_score_test = feature_importance_mask(local_fi_score_test, local_fi_score_test, mode, mask_to = "zero")
        local_fi_score_test_subset = feature_importance_mask(local_fi_score_test_subset, local_fi_score_test_subset, mode, mask_to = "zero")
        return local_fi_score_train, np.abs(local_fi_score_train_subset), np.abs(local_fi_score_test), np.abs(local_fi_score_test_subset)


def tree_shap_evaluation_RF(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset, fit=None, mode="absolute"):
    """
    Compute average treeshap value across observations.
    Larger absolute values indicate more important features.
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest (tree-based)
    :return: dataframe of shape: (n_samples, n_features)
    """
    explainer = shap.TreeExplainer(fit)
    local_fi_score_train = None
    local_fi_score_train_subset = explainer.shap_values(X_train_subset, check_additivity=False)
    local_fi_score_test = explainer.shap_values(X_test, check_additivity=False)
    local_fi_score_test_subset = explainer.shap_values(X_test_subset, check_additivity=False)
    if sklearn.base.is_classifier(fit):
        if mode == "absolute":
            return None, np.sum(np.abs(local_fi_score_train_subset),axis=-1), np.sum(np.abs(local_fi_score_test),axis=-1), np.sum(np.abs(local_fi_score_test_subset),axis=-1)
        else:
            return None, local_fi_score_train_subset[:,:,1], local_fi_score_test[:,:,1], local_fi_score_test_subset[:,:,1]
    else:
        if mode == "absolute":
            return None, np.abs(local_fi_score_train_subset), np.abs(local_fi_score_test), np.abs(local_fi_score_test_subset)
        else:
            local_fi_score_train_subset = feature_importance_mask(local_fi_score_train_subset, local_fi_score_train_subset, mode, mask_to = "zero")
            local_fi_score_test = feature_importance_mask(local_fi_score_test, local_fi_score_test, mode, mask_to = "zero")
            local_fi_score_test_subset = feature_importance_mask(local_fi_score_test_subset, local_fi_score_test_subset, mode, mask_to = "zero")
            return local_fi_score_train, np.abs(local_fi_score_train_subset), np.abs(local_fi_score_test), np.abs(local_fi_score_test_subset)
        

def kernel_shap_evaluation_RF_plus(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset, fit=None, mode="absolute"):
    assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
    rf_plus_kernel_shap = RFPlusKernelSHAP(fit)
    local_fi_score_train = None
    local_fi_score_train_subset = rf_plus_kernel_shap.explain(X_train=X_train, X_test=X_train_subset)
    local_fi_score_test = None
    local_fi_score_test_subset = rf_plus_kernel_shap.explain(X_train=X_train, X_test=X_test_subset)
    if sklearn.base.is_classifier(fit):
        if mode == "absolute":
            return None, np.sum(np.abs(local_fi_score_train_subset),axis=-1), None, np.sum(np.abs(local_fi_score_test_subset),axis=-1)
        else:
            return None, local_fi_score_train_subset[:,:,1], None, local_fi_score_test_subset[:,:,1]
    else:
        if mode == "absolute":
            return None, np.abs(local_fi_score_train_subset), None, np.abs(local_fi_score_test_subset)
        else:
            local_fi_score_train_subset = feature_importance_mask(local_fi_score_train_subset, local_fi_score_train_subset, mode, mask_to = "zero")
            local_fi_score_test_subset = feature_importance_mask(local_fi_score_test_subset, local_fi_score_test_subset, mode, mask_to = "zero")
            return local_fi_score_train, np.abs(local_fi_score_train_subset), local_fi_score_test, np.abs(local_fi_score_test_subset)
        

def lime_evaluation_RF_plus(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset, fit=None, mode="absolute"):
    assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
    rf_plus_lime = RFPlusLime(fit)
    local_fi_score_train = None
    local_fi_score_train_subset = rf_plus_lime.explain(X_train=X_train, X_test=X_train_subset).values
    local_fi_score_test = None
    local_fi_score_test_subset = rf_plus_lime.explain(X_train=X_train, X_test=X_test_subset).values
    if mode == "absolute":
        return None, np.abs(local_fi_score_train_subset), None, np.abs(local_fi_score_test_subset)
    else:
        local_fi_score_train_subset = feature_importance_mask(local_fi_score_train_subset, local_fi_score_train_subset, mode, mask_to = "zero")
        local_fi_score_test_subset = feature_importance_mask(local_fi_score_test_subset, local_fi_score_test_subset, mode, mask_to = "zero")
        return local_fi_score_train, np.abs(local_fi_score_train_subset), local_fi_score_test, np.abs(local_fi_score_test_subset)

### Feature Importance Methods for RF+
def LFI_evaluation_RFPlus_inbag(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset, fit=None, mode="absolute"):
    assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
    rf_plus_mdi_train = RFPlusMDI(fit, evaluate_on="inbag")
    rf_plus_mdi_test = RFPlusMDI(fit, evaluate_on="all")
    local_fi_score_train = np.abs(rf_plus_mdi_train.explain(X=X_train, y=y_train)[0])
    local_fi_score_train_subset = None
    local_fi_score_test = np.abs(rf_plus_mdi_test.explain(X=X_test, y=None)[0])
    local_fi_score_test_subset = np.abs(rf_plus_mdi_test.explain(X=X_test_subset, y=None)[0])
    if mode != "absolute":
        local_fi_score_train_mask = rf_plus_mdi_train.explain_subtract_intercept(X=X_train, y=y_train)
        local_fi_score_test_mask = rf_plus_mdi_test.explain_subtract_intercept(X=X_test, y=None)
        local_fi_score_test_subset_mask = rf_plus_mdi_test.explain_subtract_intercept(X=X_test_subset, y=None)
        local_fi_score_train = feature_importance_mask(local_fi_score_train, local_fi_score_train_mask, mode, mask_to = "inf")
        local_fi_score_test = feature_importance_mask(local_fi_score_test, local_fi_score_test_mask, mode, mask_to = "inf")
        local_fi_score_test_subset = feature_importance_mask(local_fi_score_test_subset, local_fi_score_test_subset_mask, mode, mask_to = "inf")
    return local_fi_score_train, local_fi_score_train_subset, local_fi_score_test, local_fi_score_test_subset


def LFI_evaluation_RFPlus_oob(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset,  fit=None, mode="absolute"):
    assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
    rf_plus_mdi_train = AloRFPlusMDI(fit, evaluate_on="oob")
    rf_plus_mdi_test = AloRFPlusMDI(fit, evaluate_on="all")
    local_fi_score_train = np.abs(rf_plus_mdi_train.explain(X=X_train, y=y_train)[0])
    local_fi_score_train_subset = None
    local_fi_score_test = np.abs(rf_plus_mdi_test.explain(X=X_test, y=None)[0])
    local_fi_score_test_subset = np.abs(rf_plus_mdi_test.explain(X=X_test_subset, y=None)[0])
    if mode != "absolute":
        local_fi_score_train_mask = rf_plus_mdi_train.explain_subtract_intercept(X=X_train, y=y_train)
        local_fi_score_test_mask = rf_plus_mdi_test.explain_subtract_intercept(X=X_test, y=None)
        local_fi_score_test_subset_mask = rf_plus_mdi_test.explain_subtract_intercept(X=X_test_subset, y=None)
        local_fi_score_train = feature_importance_mask(local_fi_score_train, local_fi_score_train_mask, mode, mask_to = "inf")
        local_fi_score_test = feature_importance_mask(local_fi_score_test, local_fi_score_test_mask, mode, mask_to = "inf")
        local_fi_score_test_subset = feature_importance_mask(local_fi_score_test_subset, local_fi_score_test_subset_mask, mode, mask_to = "inf")
    return local_fi_score_train, local_fi_score_train_subset, local_fi_score_test, local_fi_score_test_subset


def LFI_evaluation_RFPlus_all(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset,  fit=None, mode="absolute"):
    assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
    rf_plus_mdi = AloRFPlusMDI(fit, evaluate_on="all")
    local_fi_score_train = np.abs(rf_plus_mdi.explain(X=X_train, y=y_train)[0])
    local_fi_score_train_subset = None
    local_fi_score_test = np.abs(rf_plus_mdi.explain(X=X_test, y=None)[0])
    local_fi_score_test_subset = np.abs(rf_plus_mdi.explain(X=X_test_subset, y=None)[0])
    if mode != "absolute":
        local_fi_score_train_mask = rf_plus_mdi.explain_subtract_intercept(X=X_train, y=y_train)
        local_fi_score_test_mask = rf_plus_mdi.explain_subtract_intercept(X=X_test, y=None)
        local_fi_score_test_subset_mask = rf_plus_mdi.explain_subtract_intercept(X=X_test_subset, y=None)
        local_fi_score_train = feature_importance_mask(local_fi_score_train, local_fi_score_train_mask, mode, mask_to = "inf")
        local_fi_score_test = feature_importance_mask(local_fi_score_test, local_fi_score_test_mask, mode, mask_to = "inf")
        local_fi_score_test_subset = feature_importance_mask(local_fi_score_test_subset, local_fi_score_test_subset_mask, mode, mask_to = "inf")
    return local_fi_score_train, local_fi_score_train_subset, local_fi_score_test, local_fi_score_test_subset


### No intercept
def LFI_evaluation_RFPlus_inbag_subtract_intercept(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset, fit=None, mode="absolute"):
    assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
    assert mode == "absolute"
    rf_plus_mdi_train = RFPlusMDI(fit, evaluate_on="inbag")
    rf_plus_mdi_test = RFPlusMDI(fit, evaluate_on="all")
    local_fi_score_train = np.abs(rf_plus_mdi_train.explain_subtract_intercept(X=X_train, y=y_train))
    local_fi_score_train_subset = None
    local_fi_score_test = np.abs(rf_plus_mdi_test.explain_subtract_intercept(X=X_test, y=None))
    local_fi_score_test_subset = np.abs(rf_plus_mdi_test.explain_subtract_intercept(X=X_test_subset, y=None))
    return local_fi_score_train, local_fi_score_train_subset, local_fi_score_test, local_fi_score_test_subset


def LFI_evaluation_RFPlus_oob_subtract_intercept(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset,  fit=None, mode="absolute"):
    assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
    assert mode == "absolute"
    rf_plus_mdi_train = AloRFPlusMDI(fit, evaluate_on="oob")
    rf_plus_mdi_test = AloRFPlusMDI(fit, evaluate_on="all")
    local_fi_score_train = np.abs(rf_plus_mdi_train.explain_subtract_intercept(X=X_train, y=y_train))
    local_fi_score_train_subset = None
    local_fi_score_test = np.abs(rf_plus_mdi_test.explain_subtract_intercept(X=X_test, y=None))
    local_fi_score_test_subset = np.abs(rf_plus_mdi_test.explain_subtract_intercept(X=X_test_subset, y=None))
    return local_fi_score_train, local_fi_score_train_subset, local_fi_score_test, local_fi_score_test_subset



def LFI_evaluation_RFPlus_all_subtract_intercept(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset,  fit=None, mode="absolute"):
    assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
    assert mode == "absolute"
    rf_plus_mdi = AloRFPlusMDI(fit, evaluate_on="all")
    local_fi_score_train = np.abs(rf_plus_mdi.explain_subtract_intercept(X=X_train, y=y_train))
    local_fi_score_train_subset = None
    local_fi_score_test = np.abs(rf_plus_mdi.explain_subtract_intercept(X=X_test, y=None))
    local_fi_score_test_subset = np.abs(rf_plus_mdi.explain_subtract_intercept(X=X_test_subset, y=None))
    return local_fi_score_train, local_fi_score_train_subset, local_fi_score_test, local_fi_score_test_subset





###
def LFI_evaluation_RFPlus_inbag_2(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset, fit=None, mode="absolute"):
    assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
    assert mode == "absolute"
    rf_plus_mdi_train = RFPlusMDI(fit, evaluate_on="inbag")
    rf_plus_mdi_test = RFPlusMDI(fit, evaluate_on="all")
    local_fi_score_train = np.abs(rf_plus_mdi_train.explain(X=X_train, y=y_train)[1])
    local_fi_score_train_subset = None
    local_fi_score_test = np.abs(rf_plus_mdi_test.explain(X=X_test, y=None)[1])
    local_fi_score_test_subset = np.abs(rf_plus_mdi_test.explain(X=X_test_subset, y=None)[1])
    return local_fi_score_train, local_fi_score_train_subset, local_fi_score_test, local_fi_score_test_subset


def LFI_evaluation_RFPlus_oob_2(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset,  fit=None, mode="absolute"):
    assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
    assert mode == "absolute"
    rf_plus_mdi_train = AloRFPlusMDI(fit, evaluate_on="oob")
    rf_plus_mdi_test = AloRFPlusMDI(fit, evaluate_on="all")
    local_fi_score_train = np.abs(rf_plus_mdi_train.explain(X=X_train, y=y_train)[1])
    local_fi_score_train_subset = None
    local_fi_score_test = np.abs(rf_plus_mdi_test.explain(X=X_test, y=None)[1])
    local_fi_score_test_subset = np.abs(rf_plus_mdi_test.explain(X=X_test_subset, y=None)[1])
    return local_fi_score_train, local_fi_score_train_subset, local_fi_score_test, local_fi_score_test_subset



def LFI_evaluation_RFPlus_all_2(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset,  fit=None, mode="absolute"):
    assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
    assert mode == "absolute"
    rf_plus_mdi = AloRFPlusMDI(fit, evaluate_on="all")
    local_fi_score_train = np.abs(rf_plus_mdi.explain(X=X_train, y=y_train)[1])
    local_fi_score_train_subset = None
    local_fi_score_test = np.abs(rf_plus_mdi.explain(X=X_test, y=None)[1])
    local_fi_score_test_subset = np.abs(rf_plus_mdi.explain(X=X_test_subset, y=None)[1])
    return local_fi_score_train, local_fi_score_train_subset, local_fi_score_test, local_fi_score_test_subset


# def LFI_evaluation_oracle_RF_plus(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset, fit=None, mode="absolute"):
#     assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
#     subsets = [(None, None),(None, None),(X_test, y_test), (X_test_subset, y_test_subset)]
#     result_tables = []
#     rf_plus_mdi = RFPlusMDI(fit, evaluate_on="all")

#     for X_data, y_data in subsets:
#         if isinstance(X_data, np.ndarray):
#             local_feature_importances, partial_preds = rf_plus_mdi.explain(X=X_data, y=y_data)
#             abs_local_feature_importances = np.abs(local_feature_importances)
#             result_tables.append(abs_local_feature_importances)
#         else:
#             result_tables.append(None)

#     return tuple(result_tables)

# def fast_r2_score(y_true, y_pred, multiclass=False):
#     """
#     Evaluates the r-squared value between the observed and estimated responses.
#     Equivalent to sklearn.metrics.r2_score but without the robust error
#     checking, thus leading to a much faster implementation (at the cost of
#     this error checking). For multi-class responses, returns the mean
#     r-squared value across each column in the response matrix.

#     Parameters
#     ----------
#     y_true: array-like of shape (n_samples, n_targets)
#         Observed responses.
#     y_pred: array-like of shape (n_samples, n_targets)
#         Predicted responses.
#     multiclass: bool
#         Whether or not the responses are multi-class.

#     Returns
#     -------
#     Scalar quantity, measuring the r-squared value.
#     """
#     numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
#     denominator = ((y_true - np.mean(y_true, axis=0)) ** 2). \
#         sum(axis=0, dtype=np.float64)
#     if multiclass:
#         return np.mean(1 - numerator / denominator)
#     else:
#         return 1 - numerator / denominator


# def neg_log_loss(y_true, y_pred):
#     """
#     Evaluates the negative log-loss between the observed and
#     predicted responses.

#     Parameters
#     ----------
#     y_true: array-like of shape (n_samples, n_targets)
#         Observed responses.
#     y_pred: array-like of shape (n_samples, n_targets)
#         Predicted probabilies.

#     Returns
#     -------
#     Scalar quantity, measuring the negative log-loss value.
#     """
#     return -log_loss(y_true, y_pred)

# def neg_mae(y_true, y_pred):
#     return -mean_absolute_error(y_true, y_pred)

# def partial_preds_to_scores(partial_preds, y_test, scoring_fn):
#     scores = []
#     for k in range(partial_preds.shape[1]):
#         y_pred = partial_preds[:,k]
#         scores.append(scoring_fn(y_test, y_pred))
#     return scores

# def LFI_global_MDI_plus_RF_Plus(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset, fit=None):
#     if isinstance(fit, RandomForestPlusRegressor):
#         scoring_fn = fast_r2_score
#     elif isinstance(fit, RandomForestPlusClassifier):
#         scoring_fn = neg_log_loss
#     test_classification_scoring_fn = neg_mae
#     y_test_subset_hat = fit.predict(X_test_subset)
#     y_test_hat = fit.predict(X_test)
#     subsets = [(X_train, y_train, y_train),(None, None, None),(X_test, None, y_test_hat), (X_test_subset, None, y_test_subset_hat)]
#     result_tables = []
#     rf_plus_mdi = AloRFPlusMDI(fit, evaluate_on="all")

#     for X_data, y_data, y_hat in subsets:
#         if isinstance(X_data, np.ndarray):
#             if isinstance(fit, RandomForestPlusClassifier) and (np.array_equal(X_data, X_test) or np.array_equal(X_data, X_test_subset)):
#                     local_feature_importances, partial_preds = rf_plus_mdi.explain(X=X_data, y=y_data)
#                     scores = partial_preds_to_scores(partial_preds, y_hat, test_classification_scoring_fn)
#                     result_tables.append(np.tile(scores, (X_data.shape[0], 1)))
#             else:
#                 local_feature_importances, partial_preds = rf_plus_mdi.explain(X=X_data, y=y_data)
#                 scores = partial_preds_to_scores(partial_preds, y_hat, scoring_fn)
#                 result_tables.append(np.tile(scores, (X_data.shape[0], 1)))
#         else:
#             result_tables.append(None)

#     return tuple(result_tables)



# ########## Pos_Neg
# # Feature Importance Methods for RF
# def tree_shap_evaluation_RF_pos_neg(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset, fit=None):
#     """
#     Compute average treeshap value across observations.
#     Larger absolute values indicate more important features.
#     :param X: design matrix
#     :param y: response
#     :param fit: fitted model of interest (tree-based)
#     :return: dataframe of shape: (n_samples, n_features)
#     """
#     def add_abs(a, b):
#         return abs(a) + abs(b)

#     subsets = [(None, None), (X_train_subset, None), (X_test, None), (X_test_subset, None)]
#     result_tables = []

#     explainer = shap.TreeExplainer(fit)

#     for X_data, _ in subsets:
#         if isinstance(X_data, np.ndarray):
#             shap_values = explainer.shap_values(X_data, check_additivity=False)
#             if sklearn.base.is_classifier(fit):
#                 # Shape values are returned as a list of arrays, one for each class
#                 #results = np.sum(np.abs(shap_values), axis=-1)
#                 results = np.sum(shap_values, axis=-1)
#             else:
#                 results = shap_values

#             result_tables.append(results)
#             result_tables.append(np.abs(results))
#         else:
#             result_tables.append(None)
#             result_tables.append(None)
#     return tuple(result_tables)

# # Feature Importance Methods for RF+
# def LFI_evaluation_RFPlus_inbag_pos_neg(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset, fit=None):
#     assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
#     subsets = [(X_train, y_train), (None, None), (X_test, None), (X_test_subset, None)]
#     result_tables = []

#     for X_data, y_data in subsets:
#         if isinstance(X_data, np.ndarray):
#             if np.array_equal(X_data, X_train):
#                 rf_plus_mdi = RFPlusMDI(fit, evaluate_on="inbag")
#             else:
#                 rf_plus_mdi = RFPlusMDI(fit, evaluate_on="all")
#             partial_preds_subtract_intercept = rf_plus_mdi.explain_subtract_intercept(X=X_data, y=y_data)
#             local_feature_importances, _ = rf_plus_mdi.explain(X=X_data, y=y_data)
#             abs_local_feature_importances = np.abs(local_feature_importances)
#             result_tables.append(partial_preds_subtract_intercept)
#             result_tables.append(abs_local_feature_importances)
#         else:
#             result_tables.append(None)
#             result_tables.append(None)
#     return tuple(result_tables)

# def LFI_evaluation_RFPlus_oob_pos_neg(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset,  fit=None):
#     assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
#     subsets = [(X_train, y_train), (None, None), (X_test, None), (X_test_subset, None)]
#     result_tables = []

#     for X_data, y_data in subsets:
#         if isinstance(X_data, np.ndarray):
#             if np.array_equal(X_data, X_train):
#                 rf_plus_mdi = AloRFPlusMDI(fit, evaluate_on="oob")
#             else:
#                 rf_plus_mdi = AloRFPlusMDI(fit, evaluate_on="all")
#             partial_preds_subtract_intercept = rf_plus_mdi.explain_subtract_intercept(X=X_data, y=y_data)
#             local_feature_importances, _ = rf_plus_mdi.explain(X=X_data, y=y_data)
#             abs_local_feature_importances = np.abs(local_feature_importances)
#             result_tables.append(partial_preds_subtract_intercept)
#             result_tables.append(abs_local_feature_importances)
#         else:
#             result_tables.append(None)
#             result_tables.append(None)
#     return tuple(result_tables)

# def LFI_evaluation_RFPlus_all_pos_neg(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset,  fit=None):
#     assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
#     subsets = [(X_train, y_train), (None, None), (X_test, None), (X_test_subset, None)]
#     result_tables = []
#     rf_plus_mdi = AloRFPlusMDI(fit, evaluate_on="all")

#     for X_data, y_data in subsets:
#         if isinstance(X_data, np.ndarray):
#             partial_preds_subtract_intercept = rf_plus_mdi.explain_subtract_intercept(X=X_data, y=y_data)
#             local_feature_importances, _ = rf_plus_mdi.explain(X=X_data, y=y_data)
#             abs_local_feature_importances = np.abs(local_feature_importances)
#             result_tables.append(partial_preds_subtract_intercept)
#             result_tables.append(abs_local_feature_importances)
#         else:
#             result_tables.append(None)
#             result_tables.append(None)

#     return tuple(result_tables)

# def lime_evaluation_RF_plus_pos_neg(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset, fit=None):
#     assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
#     subsets = [(None, None), (X_train_subset, None), (None, None), (X_test_subset, None)]
#     result_tables = []

#     for X_data, _ in subsets:
#         if isinstance(X_data, np.ndarray):
#             rf_plus_lime = RFPlusLime(fit)
#             lime_values = rf_plus_lime.explain(X_train=X_train, X_test=X_data).values
#             result_tables.append(lime_values)
#             result_tables.append(np.abs(lime_values))
#         else:
#             result_tables.append(None)
#             result_tables.append(None)

#     return tuple(result_tables)


# def kernel_shap_evaluation_RF_plus_pos_neg(X_train, y_train, X_train_subset, y_train_subset, X_test, y_test, X_test_subset, y_test_subset, fit=None):
#     assert isinstance(fit, RandomForestPlusRegressor) or isinstance(fit, RandomForestPlusClassifier)
#     subsets = [(None, None), (X_train_subset, None), (None, None), (X_test_subset, None)]
#     result_tables = []

#     for X_data, _ in subsets:
#         if isinstance(X_data, np.ndarray):
#             rf_plus_kernel_shap = RFPlusKernelSHAP(fit)
#             kernel_shap_scores = rf_plus_kernel_shap.explain(X_train=X_train, X_test=X_data)
#             result_tables.append(kernel_shap_scores)
#             result_tables.append(np.abs(kernel_shap_scores))
#         else:
#             result_tables.append(None)
#             result_tables.append(None)

#     return tuple(result_tables)
















# result_table = pd.DataFrame(kernel_shap_scores, columns=[f'Feature_{i}' for i in range(num_features)])
# result_tables.append(result_table)

# def MDI_local_sub_stumps(X, y, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
#     """
#     Compute local MDI importance for each feature and sample.
#     :param X: design matrix
#     :param y: response
#     :param fit: fitted model of interest (tree-based)
#     :return: dataframe of shape: (n_samples, n_features)

#     """
#     num_samples, num_features = X.shape
#     if isinstance(fit, RegressorMixin):
#         RFPlus = RandomForestPlusRegressor
#     elif isinstance(fit, ClassifierMixin):
#         RFPlus = RandomForestPlusClassifier
#     else:
#         raise ValueError("Unknown task.")
#     rf_plus_model = RFPlus(rf_model=fit, **kwargs)
#     rf_plus_model.fit(X, y)

#     try:
#         mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X, y=y, local_scoring_fns=mean_squared_error, version = "sub", lfi=False)["local"].values
#         if return_stability_scores:
#             raise NotImplementedError
#             stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
#     except ValueError as e:
#         if str(e) == 'Transformer representation was empty for all trees.':
#             mdi_plus_scores = np.zeros((num_samples, num_features)) 
#             stability_scores = None
#         else:
#             raise
#     result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

#     return result_table

# def MDI_local_all_stumps(X, y, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
#     """
#     Wrapper around MDI+ object to get feature importance scores
    
#     :param X: ndarray of shape (n_samples, n_features)
#         The covariate matrix. If a pd.DataFrame object is supplied, then
#         the column names are used in the output
#     :param y: ndarray of shape (n_samples, n_targets)
#         The observed responses.
#     :param rf_model: scikit-learn random forest object or None
#         The RF model to be used for interpretation. If None, then a new
#         RandomForestRegressor or RandomForestClassifier is instantiated.
#     :param kwargs: additional arguments to pass to
#         RandomForestPlusRegressor or RandomForestPlusClassifier class.
#     :return: dataframe - [Var, Importance]
#                          Var: variable name
#                          Importance: MDI+ score
#     """
#     num_samples, num_features = X.shape
#     if isinstance(fit, RegressorMixin):
#         RFPlus = RandomForestPlusRegressor
#     elif isinstance(fit, ClassifierMixin):
#         RFPlus = RandomForestPlusClassifier
#     else:
#         raise ValueError("Unknown task.")
#     rf_plus_model = RFPlus(rf_model=fit, **kwargs)
#     rf_plus_model.fit(X, y)

#     try:
#         mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X, y=y, local_scoring_fns=mean_squared_error, version = "all", lfi=False)["local"].values
#         if return_stability_scores:
#             raise NotImplementedError
#             stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
#     except ValueError as e:
#         if str(e) == 'Transformer representation was empty for all trees.':
#             mdi_plus_scores = np.zeros((num_samples, num_features)) 
#             stability_scores = None
#         else:
#             raise
#     result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

#     return result_table


# def LFI_absolute_sum(X, y, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
#     num_samples, num_features = X.shape
#     if isinstance(fit, RegressorMixin):
#         RFPlus = RandomForestPlusRegressor
#     elif isinstance(fit, ClassifierMixin):
#         RFPlus = RandomForestPlusClassifier
#     else:
#         raise ValueError("Unknown task.")
#     rf_plus_model = RFPlus(rf_model=fit, **kwargs)
#     rf_plus_model.fit(X, y)

#     try:
#         mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X, y=y, lfi=True, lfi_abs="outside")["lfi"].values
#         if return_stability_scores:
#             raise NotImplementedError
#             stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
#     except ValueError as e:
#         if str(e) == 'Transformer representation was empty for all trees.':
#             mdi_plus_scores = np.zeros((num_samples, num_features)) 
#             stability_scores = None
#         else:
#             raise
#     result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

#     return result_table

# def lime_local(X, y, fit):
#     """
#     Compute LIME local importance for each feature and sample.
#     Larger values indicate more important features.
#     :param X: design matrix
#     :param y: response
#     :param fit: fitted model of interest (tree-based)
#     :return: dataframe of shape: (n_samples, n_features)

#     """

#     np.random.seed(1)
#     num_samples, num_features = X.shape
#     result = np.zeros((num_samples, num_features))
#     explainer = lime.lime_tabular.LimeTabularExplainer(X, verbose=False, mode='regression')
#     for i in range(num_samples):
#         exp = explainer.explain_instance(X[i], fit.predict, num_features=num_features)
#         original_feature_importance = exp.as_map()[1]
#         sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
#         for j in range(num_features):
#             result[i,j] = abs(sorted_feature_importance[j][1])
#     # Convert the array to a DataFrame
#     result_table = pd.DataFrame(result, columns=[f'Feature_{i}' for i in range(num_features)])

#     return result_table

# def tree_shap_local(X, y, fit):
#     """
#     Compute average treeshap value across observations.
#     Larger absolute values indicate more important features.
#     :param X: design matrix
#     :param y: response
#     :param fit: fitted model of interest (tree-based)
#     :return: dataframe of shape: (n_samples, n_features)
#     """
#     explainer = shap.TreeExplainer(fit)
#     shap_values = explainer.shap_values(X, check_additivity=False)
#     if sklearn.base.is_classifier(fit):
#         # Shape values are returned as a list of arrays, one for each class
#         def add_abs(a, b):
#             return abs(a) + abs(b)
#         results = np.sum(np.abs(shap_values),axis=-1)
#     else:
#         results = abs(shap_values)
#     result_table = pd.DataFrame(results, columns=[f'Feature_{i}' for i in range(X.shape[1])])

#     return result_table

# def permutation_local(X, y, fit, num_permutations=100):
#     """
#     Compute local permutation importance for each feature and sample.
#     Larger values indicate more important features.
#     :param X: design matrix
#     :param y: response
#     :param fit: fitted model of interest (tree-based)
#     :num_permutations: Number of permutations for each feature (default is 100)
#     :return: dataframe of shape: (n_samples, n_features)
#     """

#     # Get the number of samples and features
#     num_samples, num_features = X.shape

#     # Initialize array to store local permutation importance
#     lpi = np.zeros((num_samples, num_features))

#     # For each feature
#     for k in range(num_features):
#         # Permute X_k num_permutations times
#         for b in range(num_permutations):
#             X_permuted = X.copy()
#             X_permuted[:, k] = np.random.permutation(X[:, k])
            
#             # Feed permuted data through the fitted model
#             y_pred_permuted = fit.predict(X_permuted)

#             # Calculate MSE for each sample
#             for i in range(num_samples):
#                 lpi[i, k] += (y[i]-y_pred_permuted[i])**2

#     lpi /= num_permutations

#     # Convert the array to a DataFrame
#     result_table = pd.DataFrame(lpi, columns=[f'Feature_{i}' for i in range(num_features)])

#     return result_table


# def MDI_local_sub_stumps_evaluate(X_train, y_train, X_test, y_test, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
#     """
#     Compute local MDI importance for each feature and sample.
#     :param X: design matrix
#     :param y: response
#     :param fit: fitted model of interest (tree-based)
#     :return: dataframe of shape: (n_samples, n_features)

#     """
#     num_samples, num_features = X_test.shape
#     if isinstance(fit, RegressorMixin):
#         RFPlus = RandomForestPlusRegressor
#     elif isinstance(fit, ClassifierMixin):
#         RFPlus = RandomForestPlusClassifier
#     else:
#         raise ValueError("Unknown task.")
#     rf_plus_model = RFPlus(rf_model=fit, **kwargs)
#     rf_plus_model.fit(X_train, y_train)

#     try:
#         mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X_test, y=y_test, local_scoring_fns=scoring_fns, version = "sub", lfi=False, sample_split=None)["local"].values
#         if return_stability_scores:
#             raise NotImplementedError
#             stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
#     except ValueError as e:
#         if str(e) == 'Transformer representation was empty for all trees.':
#             mdi_plus_scores = np.zeros((num_samples, num_features)) 
#             stability_scores = None
#         else:
#             raise
#     result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

#     return result_table

# def lime_local(X_train, y_train, X_test, y_test, fit):
#     """
#     Compute LIME local importance for each feature and sample.
#     Larger values indicate more important features.
#     :param X: design matrix
#     :param y: response
#     :param fit: fitted model of interest (tree-based)
#     :return: dataframe of shape: (n_samples, n_features)

#     """
#     if isinstance(fit, RegressorMixin):
#         mode='regression'
#     elif isinstance(fit, ClassifierMixin):
#         mode='classification'
#     np.random.seed(1)
#     num_samples, num_features = X_test.shape
#     result = np.zeros((num_samples, num_features))
#     explainer = lime.lime_tabular.LimeTabularExplainer(X_train, verbose=False, mode=mode)
#     for i in range(num_samples):
#         exp = explainer.explain_instance(X_test[i], fit.predict, num_features=num_features)
#         original_feature_importance = exp.as_map()[1]
#         sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
#         for j in range(num_features):
#             result[i,j] = abs(sorted_feature_importance[j][1])
#     # Convert the array to a DataFrame
#     result_table = pd.DataFrame(result, columns=[f'Feature_{i}' for i in range(num_features)])

#     return result_table

# def MDI_local_all_stumps_evaluate(X_train, y_train, X_test, y_test, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
#     """
#     Wrapper around MDI+ object to get feature importance scores
    
#     :param X: ndarray of shape (n_samples, n_features)
#         The covariate matrix. If a pd.DataFrame object is supplied, then
#         the column names are used in the output
#     :param y: ndarray of shape (n_samples, n_targets)
#         The observed responses.
#     :param rf_model: scikit-learn random forest object or None
#         The RF model to be used for interpretation. If None, then a new
#         RandomForestRegressor or RandomForestClassifier is instantiated.
#     :param kwargs: additional arguments to pass to
#         RandomForestPlusRegressor or RandomForestPlusClassifier class.
#     :return: dataframe - [Var, Importance]
#                          Var: variable name
#                          Importance: MDI+ score
#     """
#     num_samples, num_features = X_test.shape
#     if isinstance(fit, RegressorMixin):
#         RFPlus = RandomForestPlusRegressor
#     elif isinstance(fit, ClassifierMixin):
#         RFPlus = RandomForestPlusClassifier
#     else:
#         raise ValueError("Unknown task.")
#     rf_plus_model = RFPlus(rf_model=fit, **kwargs)
#     rf_plus_model.fit(X_train, y_train)

#     try:
#         mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X_test, y=y_test, local_scoring_fns=scoring_fns, version = "all", lfi=False, sample_split=None)["local"].values
#         if return_stability_scores:
#             raise NotImplementedError
#             stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
#     except ValueError as e:
#         if str(e) == 'Transformer representation was empty for all trees.':
#             mdi_plus_scores = np.zeros((num_samples, num_features)) 
#             stability_scores = None
#         else:
#             raise
#     result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

#     return result_table

# def LFI_ablation_test_evaluation(X_train, y_train, X_test, y_test, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
#     num_samples, num_features = X_test.shape
#     if isinstance(fit, RegressorMixin):
#         RFPlus = RandomForestPlusRegressor
#     elif isinstance(fit, ClassifierMixin):
#         RFPlus = RandomForestPlusClassifier
#     else:
#         raise ValueError("Unknown task.")
#     rf_plus_model = RFPlus(rf_model=fit, **kwargs)
#     rf_plus_model.fit(X_train, y_train)

#     try:
#         mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X_test, y=y_test, lfi=True, lfi_abs="none", sample_split=None, train_or_test = "test")["lfi"].values
#         mdi_plus_scores = np.abs(mdi_plus_scores)
#         if return_stability_scores:
#             raise NotImplementedError
#             stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
#     except ValueError as e:
#         if str(e) == 'Transformer representation was empty for all trees.':
#             mdi_plus_scores = np.zeros((num_samples, num_features)) 
#             stability_scores = None
#         else:
#             raise
#     result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

#     return result_table

######################## Considering not using these methods
# def LFI_sum_absolute_evaluate(X_train, y_train, X_test, y_test, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
#     num_samples, num_features = X_test.shape
#     if isinstance(fit, RegressorMixin):
#         RFPlus = RandomForestPlusRegressor
#     elif isinstance(fit, ClassifierMixin):
#         RFPlus = RandomForestPlusClassifier
#     else:
#         raise ValueError("Unknown task.")
#     rf_plus_model = RFPlus(rf_model=fit, **kwargs)
#     rf_plus_model.fit(X_train, y_train)

#     try:
#         mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X_test, y=y_test, lfi=True, lfi_abs="inside", sample_split=None)["lfi"].values
#         if return_stability_scores:
#             raise NotImplementedError
#             stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
#     except ValueError as e:
#         if str(e) == 'Transformer representation was empty for all trees.':
#             mdi_plus_scores = np.zeros((num_samples, num_features)) 
#             stability_scores = None
#         else:
#             raise
#     result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

#     return result_table

# def LFI_sum_absolute(X, y, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
#     num_samples, num_features = X.shape
#     if isinstance(fit, RegressorMixin):
#         RFPlus = RandomForestPlusRegressor
#     elif isinstance(fit, ClassifierMixin):
#         RFPlus = RandomForestPlusClassifier
#     else:
#         raise ValueError("Unknown task.")
#     rf_plus_model = RFPlus(rf_model=fit, **kwargs)
#     rf_plus_model.fit(X, y)

#     try:
#         mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X, y=y, lfi=True, lfi_abs="inside")["lfi"].values
#         if return_stability_scores:
#             raise NotImplementedError
#             stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
#     except ValueError as e:
#         if str(e) == 'Transformer representation was empty for all trees.':
#             mdi_plus_scores = np.zeros((num_samples, num_features)) 
#             stability_scores = None
#         else:
#             raise
#     result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

#     return result_table