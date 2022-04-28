import numpy as np
from collections import defaultdict

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import BaseEnsemble
import statistics
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
import statsmodels.stats.multitest as smt
from tqdm import tqdm
import torch
import sys, os
from torch import nn
from numpy import linalg as LA
from torch.functional import F
import copy
from sklearn.model_selection import GridSearchCV
from torch.autograd import Variable
import numpy as np
from collections import defaultdict
from joblib import delayed, Parallel

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import r2_score
import statistics
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from nonlinear_significance.scripts.util import TreeTransformer

# from nonlinear_significance.scripts.util import *
# from nonlinear_significance.scripts.util import TreeTransformer
# sys.path.append("../../nonlinear_significance/scripts/")
# from util import TreeTransformer
# os.chdir("../../nonlinear_significance/scripts/")
from util import *


def get_r_squared(OLS_results, tree_transformer, transformed_feats, y_test, origin_feat):
    feat_pcs = tree_transformer.original_feat_to_transformed_mapping[origin_feat]
    restricted_model_coeffs = OLS_results.params[feat_pcs]
    a = np.transpose(y_test - transformed_feats[:, feat_pcs] @ restricted_model_coeffs) @ (
            y_test - transformed_feats[:, feat_pcs] @ restricted_model_coeffs)
    return 1.0 - (a / (np.transpose(y_test) @ y_test))


class TreeTester:

    def __init__(self, estimator, max_components='median', normalize=True):
        self.estimator = estimator
        self.max_components = max_components
        self.normalize = normalize

    def get_feature_significance_and_ranking(self, X, y, num_splits=10, add_linear=True, joint=False):
        p_vals = np.zeros((num_splits, X.shape[1]))
        r_squared = np.zeros((num_splits, X.shape[1]))
        for i in tqdm(range(num_splits)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)  # perform sample splitting
            self.estimator.fit(X_train, y_train)  # fit on half of sample to learn tree structure and features
            if self.max_components == 'median':
                tree_transformer = TreeTransformer(estimator=self.estimator, max_components='median')
            else:
                tree_transformer = TreeTransformer(estimator=self.estimator,
                                                   max_components=int(self.max_components * X_train.shape[0]))
            tree_transformer.fit(X_train)  # Apply PCA on X_train
            transformed_feats = tree_transformer.transform(X_test)  # apply tree mapping on X_test
            if joint:  # Fit joint linear model
                if add_linear:
                    pass
                if transformed_feats.shape[1] == 0:
                    continue
                OLS_results = sm.OLS(y_test, transformed_feats).fit()  # fit decision tree on honest sample
                for j in range(X.shape[1]):
                    num_stumps = transformed_feats.shape[1]
                    num_regressors = len(tree_transformer.original_feat_to_transformed_mapping[j])
                    # print(num_regressors)#tree_transformer.original_feat_to_transformed_mapping[j]
                    # print("num stumps for feature "  + str(j) + " are: " + str(num_regressors))
                    P_j = np.zeros((num_regressors, num_stumps))  # num_regressors
                    if num_regressors == 0:
                        p_vals[i, j] = 1.0
                        r_squared[i, j] = 0.0
                    else:
                        for (row_num, feat) in enumerate(tree_transformer.original_feat_to_transformed_mapping[j]):
                            P_j[row_num, feat] = 1.0
                        p_vals[i, j] = OLS_results.wald_test(P_j,
                                                             scalar=False).pvalue  # OLS_results.wald_test(P_j,scalar = False).pvalue
                        r_squared[i, j] = get_r_squared(OLS_results, tree_transformer, transformed_feats, y_test, j)
            else:
                for j in range(X.shape[1]):  # Iterate over original features
                    if self.max_components == 'median':
                        transformed_feats_for_j = tree_transformer.get_transformed_X_for_feat(transformed_feats, j, 0)
                    else:
                        transformed_feats_for_j = tree_transformer.get_transformed_X_for_feat(transformed_feats, j,
                                                                                              self.max_components)
                    if add_linear:
                        transformed_feats_for_j = np.hstack(
                            [X_test[:, [j]] - np.mean(X_test[:, j]), transformed_feats_for_j])
                    if transformed_feats_for_j.shape[1] == 0:
                        p_vals[i, j] = 1.0
                        r_squared[i, j] = 0.0
                    else:
                        OLS_for_j = sm.OLS(y_test - np.mean(y_test), transformed_feats_for_j).fit(cov_type="HC0")
                        r_squared[i, j] = OLS_for_j.rsquared
                        p_vals[i, j] = OLS_for_j.f_pvalue

        p_vals[np.isnan(p_vals)] = 1.0
        median_p_vals = 2 * np.median(p_vals, axis=0)
        r_squared = np.mean(r_squared, axis=0)
        median_p_vals[median_p_vals > 1.0] = 1.0

        return median_p_vals, r_squared

    def multiple_testing_correction(self, p_vals, method='bonferroni', alpha=0.05):
        return smt.multipletests(p_vals, method=method)[1]


    def get_r_squared_sig_threshold(self, X, y, num_splits=10, add_linear=True, threshold=0.05, first_ns=True):
        """
        Get r squared values, but only with respect to a subset of the engineered features, depending on a thresholding
        criterion.

        :param X:
        :param y:
        :param num_splits:
        :param add_linear:
        :param threshold:
        :param first_ns: Flag, if True, then use only engineered features with indices less than the first one
            that has nonsignificant p-value
        :return:
        """
        r_squared = np.zeros((num_splits, X.shape[1]))
        for i in tqdm(range(num_splits)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)  # perform sample splitting
            self.estimator.fit(X_train, y_train)  # fit on half of sample to learn tree structure and features
            if self.max_components == 'median':
                tree_transformer = TreeTransformer(estimator=self.estimator, max_components='median')
            else:
                tree_transformer = TreeTransformer(estimator=self.estimator,
                                                   max_components=int(self.max_components * X_train.shape[0]))
            tree_transformer.fit(X_train)  # Apply PCA on X_train
            transformed_feats = tree_transformer.transform(X_test)  # apply tree mapping on X_test
            for j in range(X.shape[1]):  # Iterate over original features
                if self.max_components == 'median':
                    transformed_feats_for_j = tree_transformer.get_transformed_X_for_feat(transformed_feats, j, 0)
                else:
                    transformed_feats_for_j = tree_transformer.get_transformed_X_for_feat(transformed_feats, j,
                                                                                          self.max_components)
                if add_linear:
                    transformed_feats_for_j = np.hstack(
                        [X_test[:, [j]] - np.mean(X_test[:, j]), transformed_feats_for_j])
                if transformed_feats_for_j.shape[1] == 0:
                    r_squared[i, j] = 0.0
                else:
                    f_p_values = sequential_F_test(transformed_feats_for_j, y_test - np.mean(y_test))
                    if first_ns:
                        if np.all(f_p_values <= threshold):
                            stopping_index = transformed_feats_for_j.shape[1]
                        else:
                            stopping_index = np.nonzero(f_p_values > threshold)[0][0] # Find first index with nonsignificant p-value
                        filtered_transformed_feats_for_j = transformed_feats_for_j[:, np.arange(stopping_index)]
                    else:
                        filtered_transformed_feats_for_j = transformed_feats_for_j[:, f_p_values <= threshold]
                    if filtered_transformed_feats_for_j.shape[1] == 0:
                        r_squared[i, j] = 0.0
                    else:
                        OLS_for_j = sm.OLS(y_test - np.mean(y_test), filtered_transformed_feats_for_j).fit(cov_type="HC0")
                        r_squared[i, j] = OLS_for_j.rsquared
        r_squared = np.mean(r_squared, axis=0)

        return r_squared


class optimalTreeTester:  # This class is trying to improve the power of TreeTester by implementing an optimal weighting scheme that favors big nodes...

    def __init__(self, estimator, normalize=True):
        self.estimator = estimator
        self.normalize = normalize

    def get_feature_significance(self, X, y, num_splits=10, eta=None, lr=.1, n_steps=3000, num_reps=20000,
                                 max_components='median', params={}):
        p_vals = np.ones((num_splits, X.shape[1]))
        r_squared = np.zeros((num_splits, X.shape[1]))
        for i in tqdm(range(num_splits)):
            X_sel, X_inf, y_sel, y_inf = train_test_split(X, y, test_size=0.5)

            if len(params) != 0:
                gs_estimator = GridSearchCV(self.estimator, param_grid=params, scoring='r2', cv=5)
                gs_estimator.fit(X_sel, y_sel)
                self.estimator = gs_estimator.best_estimator_
                self.estimator.fit(X_sel, y_sel)  # fit on half of sample to learn tree structure and features
            else:
                self.estimator.fit(X_sel, y_sel)

            if max_components == 'median':
                tree_transformer_sel = TreeTransformer(estimator=copy.deepcopy(self.estimator), max_components='median')
                tree_transformer_inf = TreeTransformer(estimator=copy.deepcopy(self.estimator), max_components='median')

            else:
                tree_transformer_sel = TreeTransformer(estimator=self.estimator,
                                                       max_components=int(self.max_components * X_train.shape[0]))
                tree_transformer_inf = TreeTransformer(estimator=self.estimator,
                                                       max_components=int(self.max_components * X_train.shape[0]))

            # tree_transformer_sel = TreeTransformer(estimator = copy.deepcopy(self.estimator), max_components= int(X_sel.shape[0]*max_components) )
            tree_transformer_sel.fit(X_sel)
            transformed_feats_sel = tree_transformer_sel.transform(X_sel)
            transformed_feats_inf = tree_transformer_sel.transform(X_inf)

            # tree_transformer_inf = TreeTransformer(estimator = copy.deepcopy(self.estimator), max_components= int(X_sel.shape[0]*max_components) )
            # tree_transformer_inf.fit(X_sel)
            # transformed_feats_inf = tree_transformer_inf.transform(X_inf)#tree_transformer_sel.transform(X_inf)#tree_transformer_inf.transform(X_inf)

            n_sel = len(y_sel)
            n_inf = len(y_inf)
            p_sel = transformed_feats_sel.shape[1]
            p_inf = transformed_feats_inf.shape[1]
            for j in range(X.shape[1]):
                stumps_sel_for_feat = tree_transformer_sel.original_feat_to_transformed_mapping[j]
                num_splits_for_feat = len(stumps_sel_for_feat)
                if num_splits_for_feat == 0:
                    p_vals[i, j] = 1.0
                else:
                    stumps_inf_for_feat = tree_transformer_sel.original_feat_to_transformed_mapping[j]
                    # tree_transformer_inf.original_feat_to_transformed_mapping[j]
                    X_sel_for_feat = transformed_feats_sel[:, stumps_sel_for_feat]
                    p_sel_feat = X_sel_for_feat.shape[1]
                    sigma_sel = (np.sum((y_sel - np.mean(y_sel)) ** 2)) / (n_sel - p_sel_feat - 1)
                    if eta is None:
                        eta = sigma_sel
                    X_inf_for_feat = transformed_feats_inf[:, stumps_inf_for_feat]
                    p_inf_feat = X_inf_for_feat.shape[1]
                    sigma_inf = (np.sum((y_inf - np.mean(y_inf)) ** 2)) / (n_inf - p_inf_feat - 1)
                    optimal_lambda_for_feat = self.get_optimal_lambda(X_sel_for_feat, y_sel, eta, sigma_sel, lr,
                                                                      n_steps)
                    p_vals[i, j] = self.compute_p_val(optimal_lambda_for_feat, X_inf_for_feat, y_inf, sigma_inf,
                                                      num_reps, n_sel, n_inf, p_sel_feat, p_inf_feat)
                    OLS_results = sm.OLS(y_inf, transformed_feats_inf).fit()
                    r_squared[i, j] = get_r_squared(OLS_results, tree_transformer_sel, transformed_feats_inf, y_inf, j)

        p_vals[np.isnan(p_vals)] = 1.0
        median_p_vals = 2 * np.median(p_vals, axis=0)
        median_p_vals[median_p_vals > 1.0] = 1.0
        r_squared = np.mean(r_squared, axis=0)

        # return median_p_vals,p_vals,self.multiple_testing_correction(median_p_vals)
        return median_p_vals, r_squared

    def multiple_testing_correction(self, p_vals, method='bonferroni', alpha=0.05):
        return smt.multipletests(p_vals, method=method)[1]

    def compute_p_val(self, optimal_lambda, X_inf, y_inf, sigma_inf, num_reps, n_sel, n_inf, p_sel_feat, p_inf_feat):
        u_inf, s_inf, vh_inf = np.linalg.svd(X_inf, full_matrices=False)
        optimal_weights = optimal_lambda  # optimal_lambda.cpu().detach().numpy()
        weighted_chi_squared_samples = np.sort(
            np.array(self.get_weighted_chi_squared(optimal_weights, n_sel, n_inf, p_sel_feat, p_inf_feat, num_reps)))
        test_statistic = (np.sum((optimal_weights * (np.transpose(u_inf) @ y_inf)) ** 2)) / sigma_inf
        quantile = stats.percentileofscore(weighted_chi_squared_samples, test_statistic, 'weak') / 100.0
        return 1.0 - quantile

    def get_optimal_lambda(self, X, y, eta, sigma_sel, lr, n_steps):
        u_sel, s_sel, vh_sel = np.linalg.svd(X, full_matrices=False)
        betas = np.transpose(u_sel) @ y
        weights = []
        for i in range(u_sel.shape[1]):
            weights.append(np.random.uniform())
        weights = np.array(weights)
        difference_in_weights = np.array([1.0])
        gradient = np.array([1.0])
        num_steps = 0
        for i in range(
                n_steps):  # while any(i > 0.00001 for i in difference_in_weights):## #or i < n_steps:#for i in range(n_steps):
            gradient = self.compute_gradient(weights, betas, u_sel, y, eta, sigma_sel)
            new_weights = np.add(weights, lr * gradient)
            difference_in_weights = new_weights - weights
            weights = new_weights
            num_steps += 1
        return weights

    def compute_gradient(self, weights, betas, u_sel, y_sel, eta, sigma_sel):
        gradients = []
        g = np.dot(weights ** 2, betas ** 2) - eta * ((LA.norm(weights)) ** 2)
        h = sigma_sel * np.sqrt(np.sum(weights ** 4))
        for i in range(len(weights)):
            g_prime = 2.0 * weights[i] * (betas[i] ** 2) - (2.0 * eta * weights[i])
            h_prime = ((2.0 * (weights[i] ** 3)) * (sigma_sel)) / (np.sqrt(np.sum(weights ** 4)))
            grad_weight = (g_prime * h - h_prime * g) / (h ** 2)
            gradients.append(grad_weight)
        return np.array(gradients)

    def get_weighted_chi_squared(self, weights, n_sel, n_inf, p_sel_feat, p_inf_feat, num_reps=10000000):
        k = len(weights)
        samples = []
        for n in range(num_reps):
            numerator_sample = 0.0
            denominator_sample = np.random.chisquare(n_sel - p_sel_feat)  # /(n_sel-p_sel_feat-1)
            for i in range(k):
                numerator_sample += ((weights[i] ** 2) * np.random.chisquare(1, size=None))
            numerator_sample = numerator_sample * (n_sel - p_sel_feat - 1)
            samples.append(numerator_sample / (denominator_sample))
        return samples

        # def forward(self,weights,u_sel,y_sel,eta,sigma_sel):
    #    T1 = torch.from_numpy(np.transpose(u_sel) @ y_sel)
    #    T1 = T1.type(torch.FloatTensor)
    #    T1 = weights * T1
    #    T1 = torch.linalg.vector_norm(T1)**2
    #    T2 = eta*torch.sum(weights**2)
    #   T3 = sigma_sel*torch.sqrt(torch.sum(weights**4))
    #   return torch.divide(torch.subtract(T2,T1),T3)
    # print("g_prime" + str(g_prime))
    # print("g" + str(g))
    # print("h_prime:" + str(h_prime))
    ##print("h" + str(h))
    # print("grad weight numerator:" + str(g_prime*h - h_prime*g))
    # print("grad_weight:" + str(grad_weight))
    # print(g_prime*h - h_prime*g[0])

    # print(weights)
    # if all(i <= 0.000001 for i in difference_in_weights):
    #    #print("im'here")
    #    break
    # else:
    #    new_weights
    # opt.zero_grad()
    # z = self.forward(weights,u_sel,y,eta,sigma_sel)#torch.linalg.vector_norm(x)#x*x
    # z.sum().backward()
    # z.sum().backward()
    # print(weights.grad.data)


#            z.sum().backward() # Calculate gradients
# opt.step()
# while all(gradient)
# tns = torch.distributions.Uniform(0,1.0).sample((u_sel.shape[1],))
# weights = Variable(tns, requires_grad=True)
# opt = torch.optim.SGD([weights], lr=lr)


def sequential_F_test(X, y, cov_type="HC0"):
    """
    Takes results from a statsmodel OLS model fit, and obtain F-statistic p-values for adding each feature into
    the model.

    :param X: covariate matrix
    :param y: response vector
    :return:
    """

    d = X.shape[1]
    p_values = np.zeros(d)
    ols_full = sm.OLS(y, X[:, 0]).fit(cov_type=cov_type)
    p_values[0] = ols_full.f_pvalue
    for i in range(1, d):
        ols_restricted = ols_full
        ols_full = sm.OLS(y, X[:, np.arange(i + 1)]).fit(cov_type=cov_type)
        p_values[i] = ols_full.compare_f_test(ols_restricted)[1]

    return p_values
