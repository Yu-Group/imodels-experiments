import os
import sys
import pandas as pd
import numpy as np
import sklearn.base
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_squared_error
from functools import reduce

import shap


def tree_shap_local(X, y, fit):
    """
    Compute average treeshap value across observations
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest (tree-based)
    :return: dataframe of shape: (n_samples, n_features)
    """
    explainer = shap.TreeExplainer(fit)
    shap_values = explainer.shap_values(X, check_additivity=False)
    if sklearn.base.is_classifier(fit):
        def add_abs(a, b):
            return abs(a) + abs(b)
        results = reduce(add_abs, shap_values)
    else:
        results = abs(shap_values)
    result_table = pd.DataFrame(results, columns=[f'Feature_{i}' for i in range(X.shape[1])])
    # results = results.mean(axis=0)
    # results = pd.DataFrame(data=results, columns=['importance'])
    # # Use column names from dataframe if possible
    # if isinstance(X, pd.DataFrame):
    #     results.index = X.columns
    # results.index.name = 'var'
    # results.reset_index(inplace=True)

    return result_table

def permutation_local(X, y, fit, num_permutations=100):
    """
    Compute local permutation importance for each feature and sample.
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest (tree-based)
    :num_permutations: Number of permutations for each feature (default is 100)
    :return: dataframe of shape: (n_samples, n_features)
    """

    # Get the number of samples and features
    num_samples, num_features = X.shape

    # Initialize array to store local permutation importance
    lpi = np.zeros((num_samples, num_features))

    # For each feature
    for k in range(num_features):
        # Permute X_k num_permutations times
        for b in range(num_permutations):
            X_permuted = X.copy()
            X_permuted[:, k] = np.random.permutation(X[:, k])
            
            # Feed permuted data through the fitted model
            y_pred_permuted = fit.predict(X_permuted)

            # Calculate MSE for each sample
            mse_values = mean_squared_error(y, y_pred_permuted)

            # Store MSE values in the array
            lpi[:, k] += mse_values

        # Average MSE values across permutations for each sample
        lpi[:, k] /= num_permutations

    # Convert the array to a DataFrame
    result_table = pd.DataFrame(lpi, columns=[f'Feature_{i}' for i in range(num_features)])

    return result_table

def MDI_plus_local(X, y, fit):
    """
    Compute local MDI+ importance for each feature and sample.
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest (tree-based)
    :return: dataframe of shape: (n_samples, n_features)

    """

    ## To Do for Zach: Please add the implementation of local MDI+ below
    num_samples, num_features = X.shape


    result = None

    # Convert the array to a DataFrame
    result_table = pd.DataFrame(result, columns=[f'Feature_{i}' for i in range(num_features)])

    return result_table