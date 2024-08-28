import copy
import os
import warnings
from functools import partial
from os.path import dirname
from os.path import join as oj
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, average_precision_score, mean_absolute_error
from sklearn.preprocessing import label_binarize
from sklearn.utils._encode import _unique
from sklearn import metrics
# from imodels.importance.ppms import huber_loss

DATASET_PATH = oj(dirname(os.path.realpath(__file__)), 'data')


class ModelConfig:
    def __init__(self,
                 name: str, cls,
                 vary_param: str = None, vary_param_val: Any = None,
                 other_params: Dict[str, Any] = {},
                 model_type: str = None):
        """
        name: str
            Name of the model.
        vary_param: str
            Name of the parameter to be varied
        model_type: str
            Type of model. ID is used to pair with FIModel.
        """

        self.name = name
        self.cls = cls
        self.model_type = model_type
        self.vary_param = vary_param
        self.vary_param_val = vary_param_val
        self.kwargs = {}
        if self.vary_param is not None:
            self.kwargs[self.vary_param] = self.vary_param_val
        self.kwargs = {**self.kwargs, **other_params}

    def __repr__(self):
        return self.name


class FIModelConfig:
    def __init__(self,
                 name: str, cls, ascending = True,
                 splitting_strategy: str = None,
                 vary_param: str = None, vary_param_val: Any = None,
                 other_params: Dict[str, Any] = {},
                 base_model= None,
                 model_type: str = None):
        """
        ascending: boolean
            Whether or not feature importances should be ranked in ascending or 
            descending order. Default is True, indicating that higher feature
            importance score is more important.
        splitting_strategy: str
            See util.apply_splitting_strategy(). Common inputs are "train-test" and None
        vary_param: str
            Name of the parameter to be varied
        model_type: str
            Type of model. ID is used to pair FIModel with Model.
        """

        assert splitting_strategy in {
            'train-test', 'train-tune-test', 'train-test-lowdata', 'train-tune-test-lowdata',
            'train-test-prediction', None, 'test-300'
        }
        assert base_model in ["None", "RF", "RFPlus_default", "RFPlus_inbag", "RFPlus_oob"]

        self.name = name
        self.cls = cls
        self.ascending = ascending
        self.model_type = model_type
        self.splitting_strategy = splitting_strategy
        self.base_model = base_model
        self.vary_param = vary_param
        self.vary_param_val = vary_param_val
        self.kwargs = {}
        if self.vary_param is not None:
            self.kwargs[self.vary_param] = self.vary_param_val
        self.kwargs = {**self.kwargs, **other_params}

    def __repr__(self):
        return self.name


def tp(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    return conf_mat[1][1]


def fp(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    return conf_mat[0][1]


def neg(y_true, y_pred):
    return sum(y_pred == 0)


def pos(y_true, y_pred):
    return sum(y_pred == 1)


def specificity_score(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    return conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])


def auprc_score(y_true, y_score, multi_class="raise"):
    assert multi_class in ["raise", "ovr"]
    n_classes = len(np.unique(y_true))
    if n_classes <= 2:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    else:
        # ovr is same as multi-label
        if multi_class == "raise":
            raise ValueError("Must set multi_class='ovr' to evaluate multi-class predictions.")
        classes = _unique(y_true)
        y_true_multilabel = label_binarize(y_true, classes=classes)
        return average_precision_score(y_true_multilabel, y_score)
      
      
def auroc_score(y_true, y_score, multi_class="raise", **kwargs):
    assert multi_class in ["raise", "ovr"]
    n_classes = len(np.unique(y_true))
    if n_classes <= 2:
        return metrics.roc_auc_score(y_true, y_score, multi_class=multi_class, **kwargs)
    else:
        # ovr is same as multi-label
        if multi_class == "raise":
            raise ValueError("Must set multi_class='ovr' to evaluate multi-class predictions.")
        classes = _unique(y_true)
        y_true_multilabel = label_binarize(y_true, classes=classes)
        return metrics.roc_auc_score(y_true_multilabel, y_score, **kwargs)
      
      
def neg_mean_absolute_error(y_true, y_pred, **kwargs):
    return -mean_absolute_error(y_true, y_pred, **kwargs)
  
  
# def neg_huber_loss(y_true, y_pred, **kwargs):
#     return -huber_loss(y_true, y_pred, **kwargs)


def restricted_roc_auc_score(y_true, y_score, ignored_indices=[]):
    """
    Compute AUROC score for only a subset of the samples

    :param y_true:
    :param y_score:
    :param ignored_indices:
    :return:
    """
    n_examples = len(y_true)
    mask = [i for i in range(n_examples) if i not in ignored_indices]
    restricted_auc = metrics.roc_auc_score(np.array(y_true)[mask], np.array(y_score)[mask])
    return restricted_auc


def compute_nsg_feat_corr_w_sig_subspace(signal_features, nonsignal_features, normalize=True):

    if normalize:
        normalized_nsg_features = nonsignal_features / np.linalg.norm(nonsignal_features, axis=0)
    else:
        normalized_nsg_features = nonsignal_features

    q, r = np.linalg.qr(signal_features)
    projections = np.linalg.norm(q.T @ normalized_nsg_features, axis=0)
    # nsg_feat_ranked_by_projection = np.argsort(projections)

    return projections


def apply_splitting_strategy(X: np.ndarray,
                             y: np.ndarray,
                             splitting_strategy: str,
                             split_seed: str) -> Tuple[Any, Any, Any, Any, Any, Any]:
    if splitting_strategy in {'train-test-lowdata', 'train-tune-test-lowdata'}:
        test_size = 0.90  # X.shape[0] - X.shape[0] * 0.1
    elif splitting_strategy == "train-test":
        test_size = 0.33
    elif splitting_strategy == "test-300":
        test_size = 300
    else:
        test_size = 0.2

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size, random_state=split_seed)
    X_tune = None
    y_tune = None

    if splitting_strategy in {'train-tune-test', 'train-tune-test-lowdata'}:
        X_train, X_tune, y_train, y_tune = model_selection.train_test_split(
            X_train, y_train, test_size=0.2, random_state=split_seed)

    return X_train, X_tune, X_test, y_train, y_tune, y_test
