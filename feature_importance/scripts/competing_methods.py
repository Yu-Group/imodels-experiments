import pandas as pd
import numpy as np
import sklearn.base
from sklearn.inspection import permutation_importance
import shap,os,sys
from functools import reduce
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm

from imodels.importance import R2FExp, GeneralizedMDI, GeneralizedMDIJoint
from imodels.importance import LassoScorer, RidgeScorer,ElasticNetScorer,RobustScorer,LogisticScorer,JointRidgeScorer,JointLogisticScorer,JointRobustScorer,JointLassoScorer,JointALOElasticNetScorer,JointALOLogisticScorer
from feature_importance.scripts.mdi_oob import MDI_OOB
from feature_importance.scripts.mda import MDA
from sklearn.linear_model import RidgeCV

def get_num_splits(X,y,fit):
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

def tree_mdi(X, y, fit,include_num_splits = False):
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
    av_splits = get_num_splits(X,y,fit)
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


def r2f(X, y, fit, max_components_type="auto",alpha=0.5,scoring_type = "lasso",pca = True,
        normalize=False, random_state=None, criterion="auto",split_data = True,rank_by_p_val = False,treelet = False,
        refit=True, add_raw=True, normalize_raw = False,n_splits=10,sample_weight=None,use_noise_variance = True,):
    """
    Compute feature signficance for trees
    :param X: full X data
    :param y: full response vector
    :param fit: estimator
    :return:
    """
    if criterion == "auto":
        if scoring_type == "ridge":
            criterion == "gcv"
        else:
            criterion == "bic"

    if scoring_type == "lasso":
        scorer = LassoScorer(criterion = criterion,refit = refit)
    elif scoring_type == "ridge":
        scorer = RidgeScorer(criterion=criterion)
    elif scoring_type == "logistic":
        scorer = LogisticScorer()
    else:
        scorer = ElasticNetScorer(refit=refit)

    r2f_obj = R2FExp(fit, max_components_type=max_components_type, alpha=alpha,scorer = scorer,pca = pca,normalize_raw = normalize_raw,
                  normalize=normalize, random_state=random_state,split_data = split_data,rank_by_p_val = rank_by_p_val,treelet = treelet,
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

def gMDI(X,y,fit,normalize = False,add_raw = True,normalize_raw = False,refit = True,
         scoring_type = "lasso",criterion = "aic_c",random_state = None,sample_weight = None):
    
    if scoring_type == "lasso":
        scorer = LassoScorer(criterion = criterion)
    elif scoring_type == "ridge":
        scorer = RidgeScorer(criterion = criterion)
    elif scoring_type == "logistic":
        scorer = LogisticScorer()
    else:
        scorer = ElasticNetScorer()
    
    gMDI_obj = GeneralizedMDI(fit,scorer = scorer, normalize = normalize, add_raw = add_raw,normalize_raw = normalize_raw,refit = refit, criterion = criterion, random_state = random_state)
    r_squared_mean, _, n_stumps, n_components_chosen = gMDI_obj.get_importance_scores(X, y, sample_weight=sample_weight, diagnostics=True)

    results = pd.DataFrame(data={'importance': r_squared_mean,
                                 'n_components': n_components_chosen.mean(axis=0),
                                 'n_stumps': n_stumps.mean(axis=0)},
                           columns=['importance', 'n_components', 'n_stumps'])

    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results


def gjMDI(X,y,fit,criterion = "aic_c", normalize = False,add_raw = True,normalize_raw = False,scoring_type = "ridge",random_state = None,lasso_sample_split = False,sample_weight = None,oob = True, error_metric = "auto"):

    if error_metric == "auto":
        if scoring_type == "ridge":
            error_metric = "loocv"
        else:
            error_metric = None

    if scoring_type == "lasso":
        scorer = JointLassoScorer(sample_split = lasso_sample_split, metric=error_metric)
    elif scoring_type == "ridge":
        scorer = JointRidgeScorer(criterion = criterion,metric = error_metric)
    elif scoring_type == "logistic":
         scorer = JointLogisticScorer(metric=error_metric)
    elif scoring_type == "alo_logistic":
         scorer = JointALOLogisticScorer(metric=error_metric)
    elif scoring_type == "alo_elastic":
        scorer = JointALOElasticNetScorer(metric=error_metric)
    else:
        scorer = ElasticNetScorer()
    
    gMDI_obj = GeneralizedMDIJoint(fit,scorer = scorer, normalize = normalize, add_raw = add_raw,normalize_raw = normalize_raw,random_state = random_state,oob = oob)
    if isinstance(y[0], str):
        r_squared_mean, _, _, n_stumps, n_components_chosen = gMDI_obj.get_importance_scores(X, y, sample_weight= sample_weight,diagnostics=True)
    else:
        r_squared_mean, _, n_stumps, n_components_chosen = gMDI_obj.get_importance_scores(X, y, sample_weight= sample_weight,diagnostics=True)

    results = pd.DataFrame(data={'importance': r_squared_mean,
                                 'n_components': n_components_chosen.mean(axis=0),
                                 'n_stumps': n_stumps.mean(axis=0)},
                           columns=['importance', 'n_components', 'n_stumps'])

    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results


def tree_mda(X, y, fit, type="oob", n_repeats=10, metric="auto"):
    if metric == "auto":
        if isinstance(y[0], str):
            metric = "accuracy"
            # raise ValueError('MDA has not yet been configured for classification.')
        else:
            metric = "mse"

    results, _ = MDA(fit, X, y[:, np.newaxis], type=type, n_trials=n_repeats, metric=metric)
    results = pd.DataFrame(data=results, columns=['importance'])

    # Use column names from dataframe if possible
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results

def ridge_cv(X,y,alphas=np.logspace(-4, 3, 100)):
    
    ridge_model = RidgeCV(alphas=alphas).fit(X, y)
    results = np.abs(ridge_model.coef_)
    results = pd.DataFrame(data=results, columns=['importance'])
    
    if isinstance(X, pd.DataFrame):
        results.index = X.columns
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
