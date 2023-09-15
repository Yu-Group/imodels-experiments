import os
import sys
import pandas as pd
import numpy as np
import sklearn.base
from sklearn.base import RegressorMixin, ClassifierMixin
from sksurv.base import SurvivalAnalysisMixin
from functools import reduce

import shap
from imodels.importance.rf_plus import RandomForestPlusRegressor, \
    RandomForestPlusClassifier, RandomForestPlusSurvival
from feature_importance.scripts.mdi_oob import MDI_OOB
from feature_importance.scripts.mda import MDA


def tree_mdi_plus_ensemble(X, y, fit, scoring_fns="auto", **kwargs):
    """
    Wrapper around MDI+ object to get feature importance scores

    :param X: ndarray of shape (n_samples, n_features)
        The covariate matrix. If a pd.DataFrame object is supplied, then
        the column names are used in the output
    :param y: ndarray of shape (n_samples, n_targets)
        The observed responses.
    :param fit: scikit-learn random forest object, RandomForestPlus object, or None
        The RF(+) model to be used for interpretation. If None, then a new
        RandomForestPlus object is instantiated.
    :param scoring_fns: list of scoring functions to use for MDI+ scoring
    :param kwargs: additional arguments to pass to
        RandomForestPlusRegressor or RandomForestPlusClassifier class.
    :return: dataframe - [Var, Importance]
                         Var: variable name
                         Importance: MDI+ score
    """

    if isinstance(fit, RegressorMixin):
        RFPlus = RandomForestPlusRegressor
    elif isinstance(fit, ClassifierMixin):
        RFPlus = RandomForestPlusClassifier
    elif isinstance(fit, SurvivalAnalysisMixin):
        RFPlus = RandomForestPlusSurvival
    else:
        raise ValueError("Unknown task.")

    mdi_plus_scores_dict = {}
    for rf_plus_name, rf_plus_args in kwargs.items():
        rf_plus_model = RFPlus(rf_model=fit, **rf_plus_args)
        rf_plus_model.fit(X, y)
        try:
            mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X, y=y, scoring_fns=scoring_fns)
        except ValueError as e:
            if str(e) == 'Transformer representation was empty for all trees.':
                mdi_plus_scores = pd.DataFrame(data=np.zeros(X.shape[1]), columns=['importance'])
                if isinstance(X, pd.DataFrame):
                    mdi_plus_scores.index = X.columns
                mdi_plus_scores.index.name = 'var'
                mdi_plus_scores.reset_index(inplace=True)
            else:
                raise
        for col in mdi_plus_scores.columns:
            if col != "var":
                mdi_plus_scores = mdi_plus_scores.rename(columns={col: col + "_" + rf_plus_name})
        mdi_plus_scores_dict[rf_plus_name] = mdi_plus_scores

    mdi_plus_scores_df = pd.concat([df.set_index('var') for df in mdi_plus_scores_dict.values()], axis=1)
    mdi_plus_ranks_df = mdi_plus_scores_df.rank(ascending=False).median(axis=1)
    mdi_plus_ranks_df = pd.DataFrame(mdi_plus_ranks_df, columns=["importance"]).reset_index()

    return mdi_plus_ranks_df


def tree_mdi_plus(X, y, fit, scoring_fns="auto", refit=True, mdiplus_kwargs=None,
                  return_stability_scores=False, **kwargs):
    """
    Wrapper around MDI+ object to get feature importance scores
    
    :param X: ndarray of shape (n_samples, n_features)
        The covariate matrix. If a pd.DataFrame object is supplied, then
        the column names are used in the output
    :param y: ndarray of shape (n_samples, n_targets)
        The observed responses.
    :param fit: scikit-learn random forest object, RandomForestPlus object, or None
        The RF(+) model to be used for interpretation. If None, then a new
        RandomForestPlus object is instantiated.
    :param scoring_fns: list of scoring functions to use for MDI+ scoring
    :param refit: whether to refit the model
    :param return_stability_scores: whether to return stability scores
    :param mdiplus_kwargs: kwargs to pass to RandomForestPlus.get_mdi_plus_scores()
    :param kwargs: additional arguments to pass to RandomForestPlus* class.
    :return: dataframe - [Var, Importance]
                         Var: variable name
                         Importance: MDI+ score
    """

    if refit:
        if isinstance(fit, RegressorMixin):
            RFPlus = RandomForestPlusRegressor
        elif isinstance(fit, ClassifierMixin):
            RFPlus = RandomForestPlusClassifier
        elif isinstance(fit, SurvivalAnalysisMixin):
            RFPlus = RandomForestPlusSurvival
        else:
            raise ValueError("Unknown task.")
        rf_plus_model = RFPlus(rf_model=fit, **kwargs)
        rf_plus_model.fit(X, y)
    else:
        rf_plus_model = fit
    try:
        mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(
            X=X, y=y, scoring_fns=scoring_fns, **mdiplus_kwargs
        )
        if return_stability_scores:
            stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
    except ValueError as e:
        if str(e) == 'Transformer representation was empty for all trees.':
            mdi_plus_scores = pd.DataFrame(data=np.zeros(X.shape[1]), columns=['importance'])
            if isinstance(X, pd.DataFrame):
                mdi_plus_scores.index = X.columns
            mdi_plus_scores.index.name = 'var'
            mdi_plus_scores.reset_index(inplace=True)
            stability_scores = None
        else:
            raise
    if isinstance(rf_plus_model, SurvivalAnalysisMixin):
        mdi_plus_scores["prediction_score"] = rf_plus_model.prediction_score_["cindex_ipcw"]
    else:
        mdi_plus_scores["prediction_score"] = rf_plus_model.prediction_score_
    if return_stability_scores:
        mdi_plus_scores = pd.concat([mdi_plus_scores, stability_scores], axis=1)

    return mdi_plus_scores


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
