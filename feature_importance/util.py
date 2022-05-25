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
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

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
            'train-test', 'train-tune-test', 'train-test-lowdata', 'train-tune-test-lowdata', None}

        self.name = name
        self.cls = cls
        self.ascending = ascending
        self.model_type = model_type
        self.splitting_strategy = splitting_strategy
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


def pr_auc_score(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def apply_splitting_strategy(X: np.ndarray,
                             y: np.ndarray,
                             splitting_strategy: str,
                             split_seed: str) -> Tuple[Any, Any, Any, Any, Any, Any]:
    if splitting_strategy in {'train-test-lowdata', 'train-tune-test-lowdata'}:
        test_size = 0.90  # X.shape[0] - X.shape[0] * 0.1
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
