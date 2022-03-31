import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.inspection import permutation_importance
import knockpy as kpy
from knockpy.knockoff_filter import KnockoffFilter
import shap

from nonlinear_significance.scripts.TreeTester import TreeTester, optimalTreeTester


def lin_reg_t_test(X, y, fit):
    '''
    Extracts basic t-test results from linear regression
    Based on statsmodels regression fit
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest, ideally lin reg from statsmodel
    :return: dataframe - [Var, Importance]
                         Var: variable name
                         Importance: p-values from t-test
    '''

    # check if using statsmodels fit (better to extract p-values)
    # refit if using sklearn
    if isinstance(fit, LinearRegression):
        lin_reg = sm.OLS(y, X)
        fit = lin_reg.fit()
    if isinstance(fit, LogisticRegression):
        lin_reg = sm.Logit(y, X)
        fit = lin_reg.fit()
    results = fit.pvalues
    results = pd.DataFrame(data=results, columns=['importance'])
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results


def tree_mdi(X, y, fit):
    '''
    Extract MDI values for a given tree
    OR
    Average MDI values for a given random forest
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest
    :return: dataframe - [Var, Importance]
                         Var: variable name
                         Importance: MDI or avg MDI
    '''
    results = fit.feature_importances_
    results = pd.DataFrame(data=results, columns=['importance'])

    # Use column names from dataframe if possible
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results


def perm_importance(X, y, fit, n_repeats=10):
    '''
    Compute average permutation importance values from a given model (fit)
    Can be a regression model or tree, anything compatible with scorer
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest
    :param n_repeats: number of times to permute a feature.
    :return: dataframe - [Var, Importance]
                         Var: variable name
                         Importance: average permutation importance
    '''
    results = permutation_importance(fit, X, y, n_repeats=n_repeats, random_state=0)
    results = results.importances_mean
    results = pd.DataFrame(data=results, columns=['importance'])

    # Use column names from dataframe if possible
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results


def knockpy_swap_integral(X, y, model, fdr=0.2):
    '''
    TODO: find a way to extract importances of variables + knockoffs
    Performs knockoff filtering based on a given model (lasso, tree, etc.)
    using the swap or swap integral importances method in Giminez et al (2018)
    This is also the built-in feature statistic for random forest
    :param X: design matrix (knockpy doesn't like dataframes)
    :param y: response
    :param model: model to be used, this model should NOT be fitted
                  e.g., DecisionTreeRegressor(random_state=0)
    :return: dataframe - [Var, Rejected]
                         Var: variable name
                         Rejected: 1 if rejected 0 otherwise
    '''
    model_fstat = kpy.knockoff_stats.FeatureStatistic(model=model)
    kfilter = KnockoffFilter(ksampler='gaussian', fstat=model_fstat)
    rejections = kfilter.forward(X, y, fdr=fdr)
    results = pd.DataFrame(data=rejections, columns=['importance'])
    # Use column names from dataframe if possible
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results


def tree_shap_mean(X, y, fit):
    '''
    Compute average treeshap value across observations
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest (tree-based)
    :return: dataframe - [Var, Importance]
                         Var: variable name
                         Importance: average absolute shap value
    '''
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


def tree_feature_significance(X, y, fit, max_components=np.inf, normalize=True, num_splits=10):
    '''
    Compute feature signficance for trees
    :param X: full X data
    :param y: full response vector
    :param fit: estimator
    :param max_components: proportion of variance explained for pca
    :param normalize: whether or not to normalize
    :param num_splits: number of sample splits/repetitions
    :return:
    '''

    tree_tester = TreeTester(fit, max_components=max_components, normalize=normalize)
    median_p_vals = tree_tester.get_feature_significance(X, y, num_splits=num_splits)

    results = pd.DataFrame(data=median_p_vals, columns=['importance'])
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results


def optimal_tree_feature_significance(X, y, fit, normalize=True, num_splits=10,
                                      eta=None, lr=1.0, n_steps=3000, num_reps=10000):
    '''
    Compute feature signficance for trees
    :param X: full X data
    :param y: full response vector
    :param fit: estimator
    :param num_splits: number of sample splits/repetitions
    :param eta
    :param lr
    :param n_steps
    :param num_reps
    :return:
    '''

    tree_tester = optimalTreeTester(fit, normalize=normalize)
    median_p_vals = tree_tester.get_feature_significance(X, y, num_splits=num_splits, eta=eta,
                                                         lr=lr, n_steps=n_steps, num_reps=num_reps)

    results = pd.DataFrame(data=median_p_vals, columns=['importance'])
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results
