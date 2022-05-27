import numpy as np
import scipy as sp
import pandas as pd
import sys
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import pickle as pkl
import joblib
from collections import defaultdict

from statsmodels.api import OLS

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

sys.path.append('../..')
from nonlinear_significance.scripts.TreeTester import TreeTester
from nonlinear_significance.scripts.util import TreeTransformer

from simulations_util import *
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression,LassoLarsIC

def run_sims(reg_func, n_grid, p, params, k=0, target_func=None, c=10, n_runs=10, normalize=False):
    results = defaultdict(list)
    for n in tqdm(n_grid):
        for r in range(n_runs):
            pca_results, tree_transformer = run_sim(reg_func, n, p, params, k, target_func, c, 405+r, normalize)
            results[n].append({"pca_results": pca_results,
                               "tree_transformer": tree_transformer})
    return results

def run_sim(reg_func, n, p, params, k=0, target_func=None, c=10, random_seed=405, normalize=False):
    np.random.seed(random_seed)
    X = np.random.randn(n, p)
    y = reg_func(X, **params)

    return get_pca_results(X, y, k, target_func, c, random_seed, normalize)


def get_pca_results(X, y, k=0, target_func=None, c=10, random_seed=405, normalize=False):

    if target_func is None:
        def f(Z):
            return Z[:, k]
        target_func = f

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=random_seed)
    rf_model = RandomForestRegressor(max_features=0.33, random_state=random_seed, min_samples_leaf=5)
    rf_model.fit(X_train, y_train)
    tree_transformer = TreeTransformer(rf_model, max_components_type=c, normalize=normalize)
    tree_transformer.fit(X_train)
    lin_fit = OLS(y_val, tree_transformer.transform_one_feature(X_val, k)).fit()
    pca_var_explained = tree_transformer.pca_transformers[k].explained_variance_ratio_[:c]
    original_feat = X_val[:, k]
    target = target_func(X_val)
    engineered_feats = tree_transformer.transform_one_feature(X_val, k)

    # lasso bic
    clf = LassoLarsIC(criterion="bic", normalize=False,fit_intercept = False)
    clf.fit(tree_transformer.transform_one_feature(X_val, k),y_val - np.mean(y_val))

    y_var_explained = np.zeros(c)
    y_var_explained_normalized = np.zeros(c)
    correlation = np.zeros(c)
    correlation_w_target = np.zeros(c)
    lasso_coef = np.zeros(c)
    ols_coef = np.zeros(c)
    for i in range(c):
        pc = engineered_feats[:, i]
        correlation[i] = np.corrcoef(original_feat, pc)[0,1]
        correlation_w_target[i] = np.corrcoef(target, pc)[0,1]
        single_OLS = OLS(y_val, engineered_feats[:, i]).fit()
        y_var_explained[i] = single_OLS.rsquared
        y_var_explained_normalized[i] = frac_explainable_var(y_val, target, pc)
        lasso_coef[i] = clf.coef_[i]
        ols_coef[i] =single_OLS.params
    pca_results = pd.DataFrame({"pca_var_exp": pca_var_explained,
                                "corr_with_base_feat": correlation,
                                "corr_with_target": correlation_w_target,
                                "y_var_explained": y_var_explained,
                                "y_var_exp_norm": y_var_explained_normalized,
                                "t-statistic" : lin_fit.tvalues[:c],
                                "lasso_coef": lasso_coef,
                                "ols_coef": ols_coef})

    return pca_results, tree_transformer

def frac_explainable_var(y, target, pc):

    explainable_var = np.corrcoef(y, target)[0, 1] ** 2
    explained_var = np.corrcoef(y, pc)[0, 1] ** 2

    return explained_var / explainable_var


def make_pca_variance_plot(results, n, save=False, experiment=None):

    c = results[n][0]["pca_results"].shape[0]
    nruns = len(results[n])
    cum_variances = np.zeros((nruns, c+1))
    for j, result in enumerate(results[n]):
        cum_variances[j, :] = np.cumsum(np.concatenate([[0], result["pca_results"]["pca_var_exp"]]))
    means = cum_variances.mean(axis=0)
    stds = cum_variances.std(axis=0)
    plt.errorbar(x=range(c+1), y=means, yerr=stds)
    plt.xlabel("No. of components")
    plt.ylabel("% variance explained")
    if save:
        plt.savefig(f"plots/{experiment}_pca_variance_plot.png")
    plt.show()
    return means, stds


def make_2d_plot(tree_transformer, X_val, c_plotted=6, save=False, experiment=None):
    nrow = 2
    ncol = c_plotted // nrow
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(4*ncol, 4*nrow), constrained_layout=True)
    cmap=cm.get_cmap('plasma')
    normalizer=Normalize(-10,10)
    im=cm.ScalarMappable(norm=normalizer)
    for k in range(c_plotted):
        i = k // ncol
        j = k % ncol
        ax = axes[i, j]
        ax.scatter(X_val[:, 0], X_val[:, 1], c=tree_transformer.transform_one_feature(X_val, 0)[:,k], cmap=cmap, norm=normalizer)
        ax.set_title(f"PC {k}")
        # ax.annotate("Test", xy=(0,0))
    fig.colorbar(im, ax=axes[:, ncol-1], shrink=0.8)
    if save:
        plt.savefig(f"plots/{experiment}_PCs_2d.png")
    plt.show()

def make_plot(tree_transformer, X_val, c_plotted=6, original_feat=0, reference_feat=0, save=False, experiment=None, color=False, y_val=None):
    nrow = 2
    ncol = c_plotted // nrow
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(4*ncol, 4*nrow), constrained_layout=True)
    cmap=cm.get_cmap('viridis')
    normalizer=Normalize(-10,10)
    im=cm.ScalarMappable(norm=normalizer)
    for k in range(c_plotted):
        i = k // ncol
        j = k % ncol
        ax = axes[i, j]
        if color:
            ax.scatter(X_val[:, original_feat], tree_transformer.transform_one_feature(X_val, reference_feat)[:,k], c=y_val, cmap=cmap, norm=normalizer)
        else:
            ax.scatter(X_val[:, original_feat], tree_transformer.transform_one_feature(X_val, reference_feat)[:,k])
        ax.set_title(f"PC {k}")
    if color:
        fig.colorbar(im, ax=axes[:, ncol-1], shrink=0.8)
    if save:
        plt.savefig(f"plots/{experiment}_PCs.png")
    plt.show()


def plot_across_runs(results, X, vary_n=False, pc_no=0, original_feat=0, reference_feat=0,
                     n=200, r=0, ylim=None, save=False, experiment_name=None):
    if vary_n:
        c_plotted = len(results.keys())
    else:
        c_plotted = len(list(results.values())[0])
    nrow = 2
    ncol = c_plotted // nrow
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(4*ncol, 4*nrow), constrained_layout=True)
    for k in range(c_plotted):
        i = k // ncol
        j = k % ncol
        ax = axes[i, j]
        if vary_n:
            tree_transformer = list(results.values())[k][r]["tree_transformer"]
            pc = tree_transformer.transform_one_feature(X, reference_feat)[:, pc_no]
            ax.set_title(f"n={list(results.keys())[k]}")
        else:
            tree_transformer = results[n][k]["tree_transformer"]
            pc = tree_transformer.transform_one_feature(X, reference_feat)[:, pc_no]
            ax.set_title(f"run {k}")
        ax.scatter(X[:, original_feat], pc)
        if ylim is not None:
            ax.set_ylim(ylim)
    if save:
        plt.savefig(f"plots/{experiment_name}.png")
    plt.show()