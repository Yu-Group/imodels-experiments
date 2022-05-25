import pandas as pd
from sklearn.inspection import permutation_importance
import shap

from feature_importance.scripts.TreeTester import TreeTester
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


def tree_feature_significance(X, y, fit, type="default", max_components_type='median',
                              normalize=False, fraction_chosen=1.0, num_splits=10,
                              add_linear=True, adjusted_r2=False, joint=False, criteria="bic", refit=True,
                              threshold=0.05, first_ns=True, direction='forward'):
    """
    Compute feature signficance for trees
    :param X: full X data
    :param y: full response vector
    :param fit: estimator
    :param type: one of "default" or "stepwise"
    :param max_components_type: "median" or proportion so that proportion*n pc components are used
    :param normalize: whether or not to normalize
    :param num_splits: number of sample splits/repetitions
    :param add_linear: boolean; whether or not to add raw x when testing
    :param joint: boolean; only used if type = "default"
    :param threshold: alpha threshold; only used if type = "stepwise"
    :param first_ns: boolean; only used if type = "stepwise"
    :return:
    """

    assert type in ["default", "sequential_stepwise", "ridge", "stepwise", "pca_cv", "pca_var", "bic_sequential",
                    "bic_nonsequential", "lasso"]

    tree_tester = TreeTester(fit, max_components_type=max_components_type, normalize=normalize,
                             fraction_chosen=fraction_chosen)
    if type == "default":
        median_p_vals, r2, n_components, n_stumps = tree_tester.get_feature_significance_and_ranking(X, y,
                                                                                                     num_splits=num_splits,
                                                                                                     add_linear=add_linear,
                                                                                                     joint=joint,
                                                                                                     diagnostics=True,
                                                                                                     adjusted_r2=adjusted_r2)
    elif type == "sequential_stepwise":
        r2, n_components, n_stumps = tree_tester.get_r_squared_sig_threshold(X, y, num_splits=num_splits,
                                                                             add_linear=add_linear, threshold=threshold,
                                                                             first_ns=first_ns, diagnostics=True)
        median_p_vals = r2
    elif type == "ridge":
        r2, n_components, n_stumps = tree_tester.get_r_squared_ridge(X, y, num_splits=num_splits, add_linear=add_linear,
                                                                     diagnostics=True)
        median_p_vals = r2
    elif type == "pca_cv":
        r2, n_components, n_stumps = tree_tester.get_r_squared_pca_cv(X, y, num_splits=num_splits,
                                                                      add_linear=add_linear, diagnostics=True)
        median_p_vals = r2
    elif type == "bic_sequential":
        r2, n_components, n_stumps = tree_tester.get_r_squared_sequential_bic(X, y, num_splits=num_splits,
                                                                              add_linear=add_linear, diagnostics=True,
                                                                              adjusted_r2=adjusted_r2)
        median_p_vals = r2
    elif type == "bic_nonsequential":
        r2, n_components, n_stumps = tree_tester.get_r_squared_nonsequential_bic(X, y, num_splits=num_splits,
                                                                                 add_linear=add_linear,
                                                                                 diagnostics=True,
                                                                                 adjusted_r2=adjusted_r2,
                                                                                 direction=direction)
        median_p_vals = r2
    elif type == "pca_var":
        r2, n_components, n_stumps = tree_tester.get_r_squared_pca_var_explained(X, y, num_splits=num_splits,
                                                                                 add_linear=add_linear,
                                                                                 diagnostics=True)
        median_p_vals = r2
    elif type == "lasso":
        r2, n_components, n_stumps = tree_tester.get_r_squared_lasso(X, y, num_splits=num_splits, add_linear=add_linear,
                                                                     diagnostics=True, criteria=criteria, refit=refit)
        median_p_vals = r2
    else:
        r2, n_components, n_stumps = tree_tester.get_r_squared_stepwise_regression(X, y, num_splits=num_splits,
                                                                                   add_linear=add_linear,
                                                                                   diagnostics=True)
        median_p_vals = r2
    results = pd.DataFrame(data={'importance': r2,
                                 'n_components': n_components.mean(axis=0),
                                 'n_stumps': n_stumps.mean(axis=0)},
                           columns=['importance', 'n_components', 'n_stumps'])

    if isinstance(X, pd.DataFrame):
        results.index = X.columns
    results.index.name = 'var'
    results.reset_index(inplace=True)

    return results
