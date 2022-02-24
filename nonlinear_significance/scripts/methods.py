import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import knockpy as kpy
from knockpy.knockoff_filter import KnockoffFilter
import shap

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
    if not isinstance(fit,sm.regression.linear_model.RegressionResultsWrapper):
        lin_reg = sm.OLS(y, X)
        fit = lin_reg.fit()
    results = fit.pvalues
    results = pd.DataFrame(data=results, columns=['importance'])
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results

def tree_mdi(X,y,fit):
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

def perm_importance(X,y,fit,n_repeats = 10):
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
    results = permutation_importance(fit, X, y, n_repeats=n_repeats,random_state = 0)
    results = results.importances_mean
    results = pd.DataFrame(data=results, columns=['importance'])

    # Use column names from dataframe if possible
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results

def knockpy_swap_integral(X,y,model,fdr=0.2):
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

def tree_shap_mean(X,y,fit):
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
    results = results.mean(axis = 0)
    results = pd.DataFrame(data=results, columns=['importance'])
    # Use column names from dataframe if possible
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results
