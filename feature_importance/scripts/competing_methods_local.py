import os
import sys
import pandas as pd
import numpy as np
import sklearn.base
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_squared_error
from functools import reduce

import shap
import lime
import lime.lime_tabular
from imodels.importance.rf_plus import RandomForestPlusRegressor, RandomForestPlusClassifier


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
            for i in range(num_samples):
                lpi[i, k] += (y[i]-y_pred_permuted[i])**2

    lpi /= num_permutations

    # Convert the array to a DataFrame
    result_table = pd.DataFrame(lpi, columns=[f'Feature_{i}' for i in range(num_features)])

    return result_table

##########To Do for Zach: Please add the implementation of local MDI and MDI+ below##########
def MDI_local_sub_stumps(X, y, fit):
    """
    Compute local MDI importance for each feature and sample.
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest (tree-based)
    :return: dataframe of shape: (n_samples, n_features)

    """
    num_samples, num_features = X.shape


    result = None

    # Convert the array to a DataFrame
    result_table = pd.DataFrame(result, columns=[f'Feature_{i}' for i in range(num_features)])

    return result_table

def MDI_local_all_stumps(X, y, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
    """
    Wrapper around MDI+ object to get feature importance scores
    
    :param X: ndarray of shape (n_samples, n_features)
        The covariate matrix. If a pd.DataFrame object is supplied, then
        the column names are used in the output
    :param y: ndarray of shape (n_samples, n_targets)
        The observed responses.
    :param rf_model: scikit-learn random forest object or None
        The RF model to be used for interpretation. If None, then a new
        RandomForestRegressor or RandomForestClassifier is instantiated.
    :param kwargs: additional arguments to pass to
        RandomForestPlusRegressor or RandomForestPlusClassifier class.
    :return: dataframe - [Var, Importance]
                         Var: variable name
                         Importance: MDI+ score
    """

    if isinstance(fit, RegressorMixin):
        RFPlus = RandomForestPlusRegressor
    elif isinstance(fit, ClassifierMixin):
        RFPlus = RandomForestPlusClassifier
    else:
        raise ValueError("Unknown task.")
    rf_plus_model = RFPlus(rf_model=fit, **kwargs)
    rf_plus_model.fit(X, y)
    try:
        mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X, y=y, scoring_fns=scoring_fns)
        if return_stability_scores:
            stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
    except ValueError as e:
        if str(e) == 'Transformer representation was empty for all trees.':
            mdi_plus_scores = pd.DataFrame(data=np.zeros(X.shape[1]), columns=['importance'])
            if isinstance(X, pd.DataFrame):
                mdi_plus_scores.index = X.columns
            mdi_plus_scores.index.name = 'var'
            mdi_plus_scores.reset_index(inplace=True)
            stability_scores = None
        else:
            raise
    mdi_plus_scores["prediction_score"] = rf_plus_model.prediction_score_
    if return_stability_scores:
        mdi_plus_scores = pd.concat([mdi_plus_scores, stability_scores], axis=1)

    return mdi_plus_scores

def lime_local(X, y, fit):
    """
    Compute LIME local importance for each feature and sample.
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest (tree-based)
    :return: dataframe of shape: (n_samples, n_features)

    """

    np.random.seed(1)
    num_samples, num_features = X.shape
    result = np.zeros((num_samples, num_features))
    explainer = lime.lime_tabular.LimeTabularExplainer(X, verbose=False, mode='regression')
    for i in range(num_samples):
        exp = explainer.explain_instance(X[i], fit.predict, num_features=num_features)
        original_feature_importance = exp.as_map()[1]
        sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
        for j in range(num_features):
            result[i,j] = abs(sorted_feature_importance[j][1])
    # Convert the array to a DataFrame
    result_table = pd.DataFrame(result, columns=[f'Feature_{i}' for i in range(num_features)])

    return result_table