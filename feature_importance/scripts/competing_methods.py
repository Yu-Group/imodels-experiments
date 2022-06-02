import pandas as pd
from sklearn.inspection import permutation_importance
import shap,os,sys

from imodels.importance import R2FExp
from feature_importance.scripts.mdi_oob import MDI_OOB

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
                 normalized=True, balanced=False, demean=False, normal_fX=False):
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


def r2f(X, y, fit, max_components_type="auto", alpha=0.5,
        normalize=False, random_state=None, criterion="bic",split_data = True,
        refit=True, add_raw=True, n_splits=10, sample_weight=None,use_noise_variance = True):
    """
    Compute feature signficance for trees
    :param X: full X data
    :param y: full response vector
    :param fit: estimator
    :return:
    """

    r2f_obj = R2FExp(fit, max_components_type=max_components_type, alpha=alpha,
                  normalize=normalize, random_state=random_state,split_data = split_data,
                  criterion=criterion, refit=refit, add_raw=add_raw, n_splits=n_splits,use_noise_variance = use_noise_variance) #R2FExp

    r_squared_mean, _, n_stumps, n_components_chosen = r2f_obj.get_importance_scores(
        X, y, sample_weight=sample_weight, diagnostics=True
    )

    results = pd.DataFrame(data={'importance': r_squared_mean,
                                 'n_components': n_components_chosen.mean(axis=0),
                                 'n_stumps': n_stumps.mean(axis=0)},
                           columns=['importance', 'n_components', 'n_stumps'])

    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results
