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
import math

#import rpy2.robjects as ro
#from rpy2.robjects.packages import importr
#import rpy2.robjects.numpy2ri
#from rpy2.robjects import pandas2ri
#from rpy2.robjects.conversion import localconverter
#from boruta import BorutaPy
#rpy2.robjects.numpy2ri.activate()
#base = importr('base')
#FOCI = importr('FOCI')

from nonlinear_significance.scripts.TreeTester import TreeTester, optimalTreeTester
from nonlinear_significance.scripts.competing_methods import *


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

def lin_reg_marginal_t_test(X, y, fit=None):
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
    n_feats = X.shape[1]
    results = np.zeros(n_feats)
    if fit is None or isinstance(fit, LinearRegression):
        model = sm.OLS
    elif isinstance(fit, LogisticRegression):
        model = sm.Logit
    for i in range(n_feats):
        lin_reg = model(y, X[:, i])
        fit = lin_reg.fit()
        results[i] = fit.pvalues[0]
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

def tree_mdi_OOB(X,y,fit,type = 'oob',
                 normalized = True, balanced = False, demean=False,normal_fX = False):
    
    reshaped_y = y.reshape((len(y),1))
    results = MDI_OOB(fit, X, reshaped_y, type = type,normalized = normalized,balanced = balanced, 
            demean=demean,normal_fX = normal_fX)[0]
    results = pd.DataFrame(data=results, columns=['importance'])
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)
    
    return results

def tree_MDA(X,y,fit,type = 'oob', n_trials = 10, metric = 'mse'):
    reshaped_y = y.reshape((len(y),1))
    results = MDA(fit, X, reshaped_y,type = type, n_trials = 10, metric = 'mse')[0]
    results = pd.DataFrame(data=results, columns=['importance'])
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


def knockpy_swap_integral(X,y,model,knockoff_fdr=0.2):
    '''
    Performs knockoff filtering based on a given model (lasso, tree, etc.)
    using the swap or swap integral importances method in Giminez et al (2018)
    This is also the built-in feature statistic for random forest
    :param X: design matrix (knockpy doesn't like dataframes)
    :param y: response
    :param model: model to be used, this model should NOT be fitted
                  e.g., DecisionTreeRegressor(random_state=0)
    :param knockoff_fdr: false discovery rate for knockoffs
    :return: dataframe - [Var, fstat, importance, rejections]
                         Var: variable name
                         fstat: fstatistics computed for each variable for inference
                         importance: feature importances
                         rejections: 1 if rejected 0 otherwise
    '''
    model_fstat = kpy.knockoff_stats.FeatureStatistic(model=model)
    kfilter = KnockoffFilter(ksampler='gaussian', fstat=model_fstat, knockoff_kwargs={"choldate_warning":False})
    rejections = kfilter.forward(X, y, fdr=knockoff_fdr)
    results = pd.DataFrame(data=kfilter.W, columns=['fstat'])
    results['importance'] = kfilter.Z[0:math.floor(len(kfilter.Z)/2)]
    results['rejections'] = rejections
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


def tree_feature_significance(X, y, fit, type="default", max_components_type='median',
                              normalize=False, fraction_chosen=1.0, num_splits=10,
                              add_linear=True, adjusted_r2=False, joint=False,
                              threshold=0.05, first_ns=True, direction='forward'):
    """
    Compute feature signficance for trees
    :param X: full X data
    :param y: full response vector
    :param fit: estimator
    :param type: one of "default" or "stepwise"
    :param max_components: "median" or proportion so that proportion*n pc components are used
    :param normalize: whether or not to normalize
    :param num_splits: number of sample splits/repetitions
    :param add_linear: boolean; whether or not to add raw x when testing
    :param joint: boolean; only used if type = "default"
    :param threshold: alpha threshold; only used if type = "stepwise"
    :param first_ns: boolean; only uesd if type = "stepwise"
    :return:
    """

    assert type in ["default", "sequential_stepwise", "ridge", "stepwise", "pca_cv", "pca_var", "bic_sequential", "bic_nonsequential","lasso"]

    tree_tester = TreeTester(fit, max_components_type=max_components_type, normalize=normalize,fraction_chosen = fraction_chosen)
    if type == "default":
        median_p_vals, r2, n_components, n_stumps = tree_tester.get_feature_significance_and_ranking(X, y, num_splits=num_splits, add_linear=add_linear, joint=joint, diagnostics=True, adjusted_r2=adjusted_r2)
    elif type == "sequential_stepwise":
        r2, n_components, n_stumps = tree_tester.get_r_squared_sig_threshold(X, y, num_splits=num_splits, add_linear=add_linear, threshold=threshold, first_ns=first_ns,diagnostics=True)
        median_p_vals = r2
    elif type == "ridge":
        r2, n_components, n_stumps = tree_tester.get_r_squared_ridge(X, y, num_splits=num_splits, add_linear=add_linear, diagnostics=True)
        median_p_vals = r2
    elif type == "pca_cv":
        r2, n_components, n_stumps = tree_tester.get_r_squared_pca_cv(X, y, num_splits=num_splits, add_linear=add_linear,diagnostics=True)
        median_p_vals = r2
    elif type == "bic_sequential":
        r2, n_components, n_stumps = tree_tester.get_r_squared_sequential_bic(X, y, num_splits=num_splits, add_linear=add_linear, diagnostics=True, adjusted_r2=adjusted_r2)
        median_p_vals = r2
    elif type == "bic_nonsequential":
        r2, n_components, n_stumps = tree_tester.get_r_squared_nonsequential_bic(X, y, num_splits=num_splits, add_linear=add_linear, diagnostics=True, adjusted_r2=adjusted_r2, direction=direction)
        median_p_vals = r2
    elif type == "pca_var":
        r2, n_components, n_stumps = tree_tester.get_r_squared_pca_var_explained(X, y, num_splits=num_splits, add_linear=add_linear,diagnostics=True)
        median_p_vals = r2
    elif type == "lasso":
        r2, n_components, n_stumps = tree_tester.get_r_squared_lasso(X, y, num_splits=num_splits, add_linear=add_linear,diagnostics=True)
        median_p_vals = r2
    else:
        r2, n_components, n_stumps = tree_tester.get_r_squared_stepwise_regression(X, y, num_splits=num_splits, add_linear=add_linear, diagnostics=True)
        median_p_vals = r2
    results = pd.DataFrame(data={'importance': median_p_vals,
                                 'r2': r2,
                                 'n_components': n_components.mean(axis=0),
                                 'n_stumps': n_stumps.mean(axis=0)},
                           columns=['importance', 'r2', 'n_components', 'n_stumps'])

    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results


def optimal_tree_feature_significance(X, y, fit, normalize=True, num_splits=10,
                                      eta=None, lr=0.1, n_steps=3000, max_components = 'median',num_reps=10000):
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
    median_p_vals,r2 = tree_tester.get_feature_significance(X, y, num_splits=num_splits, eta=eta,max_components = max_components,
                                                         lr=lr, n_steps=n_steps, num_reps=num_reps)

    #results = pd.DataFrame(data=median_p_vals, columns=['importance'])
    results = pd.DataFrame(data={'importance':median_p_vals,'r2':r2}, columns=['importance','r2'])
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results

def foci_rank(X, y, fit=None, numFeatures=None, numCores = 1):
    '''
    Rank all features using FOCI, a variable selection algorithm based on the measure of conditional dependence codec
    :param X: design matrix
    :param y: response
    :param numFeatures: the number of features to go up to in selection search
    :param numCores: the number of cores to use for parallelization (recommend if n is large)
    :return: dataframe - [Var, index, codec]
                         Var: selected variable name (in order of decreasing conditional predictive power)
                         index: index of selected variable
                         codec: the corresponding conditional dependence coefficient when adding in each variable
    '''
    if numFeatures is None:
        numFeatures = X.shape[1]
    result = FOCI.foci(y, X, num_features = numFeatures,stop = False,numCores = numCores)
    with localconverter(ro.default_converter + pandas2ri.converter):
        result_pd = ro.conversion.rpy2py(result[0])
    result_pd['importance'] = result[1]
    result_pd.index = result_pd['names']
    result_pd.index.name = 'var_name'
    result_pd.drop('names', inplace=True, axis=1)
    result_pd = result_pd.rename(columns={'index': 'var'})
    result_pd['var'] = (result_pd['var'] - 1).astype(int)
    result_pd.reset_index(inplace=True)

    return result_pd

def boruta_rank(X,y,fit,verbose = 1):
    '''
    Rank all features using Boruta
    :param X: design matrix
    :param y: response
    :param fit: a supervised learning estimator, with a 'fit' method that returns the
    feature_importances_ attribute. Important features must correspond to high absolute values
    in the feature_importances_.
    e.g., RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    :param verbose: level of boruta alg output 0-2
    :return: dataframe - [Var, rank]
                         Var: variable name or index of no column names provided
                         rank: rank assigned to each feature (lower is better). Note this will only go as low
                         as 2 if no features should actually be selected based on Boruta, otherwise 1 is best
    '''
    boruta_mod = BorutaPy(fit,verbose = verbose)
    boruta_fit = boruta_mod.fit(X, y)
    results = pd.DataFrame({'rank': boruta_fit.ranking_,
                            'importance': -boruta_fit.ranking_,
                            'rejections': boruta_fit.support_.astype(int)})
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)
    return results