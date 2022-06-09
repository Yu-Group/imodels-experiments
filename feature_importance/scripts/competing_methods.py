import pandas as pd
from sklearn.inspection import permutation_importance
import shap,os,sys

from imodels.importance import R2FExp, GeneralizedMDI, GeneralizedMDIJoint
from imodels.importance import LassoScorer, RidgeScorer,ElasticNetScorer,RobustScorer,LogisticScorer,JointRidgeScorer,JointLogisticScorer,JointRobustScorer
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


def r2f(X, y, fit, max_components_type="auto", alpha=0.5,scoring_type = "lasso",pca = True,
        normalize=False, random_state=None, criterion="bic",split_data = True,rank_by_p_val = False,treelet = False, 
        refit=True, add_raw=True, normalize_raw = False,n_splits=10,sample_weight=None,use_noise_variance = True,):
    """
    Compute feature signficance for trees
    :param X: full X data
    :param y: full response vector
    :param fit: estimator
    :return:
    """
    if scoring_type == "lasso":
        scorer = LassoScorer(criterion = criterion,refit = refit)
    elif scoring_type == "ridge":
        scorer = RidgeScorer()
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

def gMDI(X,y,fit,scorer = LassoScorer(),normalize = False,add_raw = True,normalize_raw = False,refit = True,
         scoring_type = "lasso",criterion = "aic_c",random_state = None,sample_weight = None):
    
    if scoring_type == "lasso":
        scorer = LassoScorer()
    elif scoring_type == "ridge":
        scorer = RidgeScorer()
    else:
        scorer = ElasticNetScorer()
    
    gMDI_obj = GeneralizedMDI(fit,scorer = scorer, normalize = normalize, add_raw = add_raw,normalize_raw = normalize_raw, 
refit = refit, criterion = criterion, random_state = random_state)
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


def gjMDI(X,y,fit,scorer = RidgeScorer(),normalize = False,add_raw = True,normalize_raw = False,scoring_type = "ridge",random_state = None):
    
    if scoring_type == "lasso":
        scorer = LassoScorer()
    elif scoring_type == "ridge":
        scorer = RidgeScorer()
    else:
        scorer = ElasticNetScorer()
    
    gMDI_obj = GeneralizedMDIJoint(fit,scorer = scorer, normalize = normalize, add_raw = add_raw,normalize_raw = normalize_raw,random_state = random_state)
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
