import os
import sys
import pandas as pd
import numpy as np
import sklearn.base
from functools import reduce

import shap
from imodels.importance import GMDI
from feature_importance.scripts.mdi_oob import MDI_OOB
from feature_importance.scripts.mda import MDA


def GMDI_pipeline(X, y, fit, **kwargs):
    """
    Wrapper around GMDI object to get feature importance scores
    
    :param X: ndarray of shape (n_samples, n_features)
        The covariate matrix. If a pd.DataFrame object is supplied, then
        the column names are used in the output
    :param y: ndarray of shape (n_samples, n_targets)
        The observed responses.
    :param rf_model: scikit-learn random forest object or None
        The RF model to be used for interpretation. If None, then a new
        RandomForestRegressor or RandomForestClassifier is instantiated.
    :param kwargs: additional arguments to pass to GMDI class.
    :return: dataframe - [Var, Importance]
                         Var: variable name
                         Importance: GMDI score
    """

    gmdi_est = GMDI(rf_model=fit, **kwargs)
    try:
        gmdi_scores = gmdi_est.get_scores(X=X, y=y)
    except ValueError as e:
        if str(e) == 'Transformer representation was empty for all trees.':
            gmdi_scores = pd.DataFrame(data=np.zeros(X.shape[1]), columns=['importance'])
            if isinstance(X, pd.DataFrame):
                gmdi_scores.index = X.columns
            gmdi_scores.index.name = 'var'
            gmdi_scores.reset_index(inplace=True)
        else:
            raise

    return gmdi_scores


def tree_mdi(X, y, fit, include_num_splits=False):
    """
    Extract MDI values for a given tree
    OR
    Average MDI values for a given random forest
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest
    :return: dataframe - [Var, Importance]
                         Var: variable name
                         Importance: MDI or avg MDI
    """
    av_splits = get_num_splits(X, y, fit)
    results = fit.feature_importances_
    results = pd.DataFrame(data=results, columns=['importance'])

    # Use column names from dataframe if possible
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    if include_num_splits:
        results['av_splits'] = av_splits
    
    return results


def tree_mdi_OOB(X, y, fit, type='oob',
                 normalized=False, balanced=False, demean=False, normal_fX=False):
    """
    Compute MDI-oob feature importance for a given random forest
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest
    :return: dataframe - [Var, Importance]
                         Var: variable name
                         Importance: MDI-oob
    """
    reshaped_y = y.reshape((len(y), 1))
    results = MDI_OOB(fit, X, reshaped_y, type=type, normalized=normalized, balanced=balanced,
                      demean=demean, normal_fX=normal_fX)[0]
    results = pd.DataFrame(data=results, columns=['importance'])
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results


def tree_shap(X, y, fit):
    """
    Compute average treeshap value across observations
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest (tree-based)
    :return: dataframe - [Var, Importance]
                         Var: variable name
                         Importance: average absolute shap value
    """
    explainer = shap.TreeExplainer(fit)
    shap_values = explainer.shap_values(X, check_additivity=False)
    if sklearn.base.is_classifier(fit):
        def add_abs(a, b):
            return abs(a) + abs(b)
        results = reduce(add_abs, shap_values)
    else:
        results = abs(shap_values)
    results = results.mean(axis=0)
    results = pd.DataFrame(data=results, columns=['importance'])
    # Use column names from dataframe if possible
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results


def tree_mda(X, y, fit, type="oob", n_repeats=10, metric="auto"):
    """
    Compute MDA importance for a given random forest
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest
    :param type: "oob" or "train"
    :param n_repeats: number of permutations
    :param metric: metric for computation MDA/permutation importance
    :return: dataframe - [Var, Importance]
                         Var: variable name
                         Importance: MDA
    """
    if metric == "auto":
        if isinstance(y[0], str):
            metric = "accuracy"
        else:
            metric = "mse"

    results, _ = MDA(fit, X, y[:, np.newaxis], type=type, n_trials=n_repeats, metric=metric)
    results = pd.DataFrame(data=results, columns=['importance'])

    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results


def get_num_splits(X, y, fit):
    """
    Gets number of splits per feature in a fitted RF 
    """
    num_splits_feature = np.zeros(X.shape[1])
    for tree in fit.estimators_:
        tree_features = tree.tree_.feature
        for i in range(X.shape[1]):
            num_splits_feature[i] += np.count_nonzero(tree_features == i)
    num_splits_feature/len(fit.estimators_)
    return num_splits_feature
