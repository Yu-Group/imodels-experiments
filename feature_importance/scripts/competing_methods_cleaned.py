import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
import shap
from feature_importance.scripts.mdi_oob import MDI_OOB

from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import OneHotEncoder

from imodels.importance.representation_cleaned import TreeTransformer, IdentityTransformer, CompositeTransformer
from imodels.importance.r2f_exp_cleaned import GMDIEnsemble, RidgeLOOPPM, LogisticLOOPPM


def tree_mdi(X, y, fit):
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
    results = fit.feature_importances_
    results = pd.DataFrame(data=results, columns=['importance'])

    # Use column names from dataframe if possible
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

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


def tree_perm_importance(X, y, fit, n_repeats=10):
    """
    Compute average permutation importance values from a given model (fit)
    Can be a regression model or tree, anything compatible with scorer
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest
    :param n_repeats: number of times to permute a feature.
    :return: dataframe - [Var, Importance]
                         Var: variable name
                         Importance: average permutation importance
    """
    results = permutation_importance(fit, X, y, n_repeats=n_repeats, random_state=0)
    results = results.importances_mean
    results = pd.DataFrame(data=results, columns=['importance'])

    # Use column names from dataframe if possible
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
    shap_values = explainer.shap_values(X)
    results = abs(shap_values)
    results = results.mean(axis=0)
    results = pd.DataFrame(data=results, columns=['importance'])
    # Use column names from dataframe if possible
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results


def GMDI_pipeline(X, y, fit, regression=True, mode="keep_k", 
                  partial_prediction_model="auto", scoring_fn="auto", 
                  include_raw=True, subsetting_scheme=None):

    p = X.shape[1]
    if include_raw:
        tree_transformers = [CompositeTransformer([TreeTransformer(p, tree_model, data=X),
                                                    IdentityTransformer(p)], adj_std="max")
                            for tree_model in fit.estimators_]
    else:
        tree_transformers = [CompositeTransformer([TreeTransformer(p, tree_model, data=X)], adj_std="max")
                            for tree_model in fit.estimators_]

    if partial_prediction_model == "auto":
        if regression:
            partial_prediction_model = RidgeLOOPPM()
        else:
            partial_prediction_model = LogisticLOOPPM()
    if scoring_fn == "auto":
        if regression:
            scoring_fn = r2_score
        else:
            scoring_fn = roc_auc_score
    if not regression:
        if len(np.unique(y)) > 2:
            y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    
    gmdi = GMDIEnsemble(tree_transformers, partial_prediction_model, scoring_fn, mode, subsetting_scheme)
    scores = gmdi.get_scores(X, y)
    
    results = pd.DataFrame(data={'importance': scores})

    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results


