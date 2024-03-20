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
from imodels.importance.rf_plus import _fast_r2_score
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score, mean_squared_error


def MDI_local_sub_stumps(X, y, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
    """
    Compute local MDI importance for each feature and sample.
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest (tree-based)
    :return: dataframe of shape: (n_samples, n_features)

    """
    num_samples, num_features = X.shape
    if isinstance(fit, RegressorMixin):
        RFPlus = RandomForestPlusRegressor
    elif isinstance(fit, ClassifierMixin):
        RFPlus = RandomForestPlusClassifier
    else:
        raise ValueError("Unknown task.")
    rf_plus_model = RFPlus(rf_model=fit, **kwargs)
    rf_plus_model.fit(X, y)

    try:
        mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X, y=y, local_scoring_fns=mean_squared_error, version = "sub", lfi=False)["local"].values
        if return_stability_scores:
            raise NotImplementedError
            stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
    except ValueError as e:
        if str(e) == 'Transformer representation was empty for all trees.':
            mdi_plus_scores = np.zeros((num_samples, num_features)) 
            stability_scores = None
        else:
            raise
    result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

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
    num_samples, num_features = X.shape
    if isinstance(fit, RegressorMixin):
        RFPlus = RandomForestPlusRegressor
    elif isinstance(fit, ClassifierMixin):
        RFPlus = RandomForestPlusClassifier
    else:
        raise ValueError("Unknown task.")
    rf_plus_model = RFPlus(rf_model=fit, **kwargs)
    rf_plus_model.fit(X, y)

    try:
        mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X, y=y, local_scoring_fns=mean_squared_error, version = "all", lfi=False)["local"].values
        if return_stability_scores:
            raise NotImplementedError
            stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
    except ValueError as e:
        if str(e) == 'Transformer representation was empty for all trees.':
            mdi_plus_scores = np.zeros((num_samples, num_features)) 
            stability_scores = None
        else:
            raise
    result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

    return result_table


def LFI_absolute_sum(X, y, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
    num_samples, num_features = X.shape
    if isinstance(fit, RegressorMixin):
        RFPlus = RandomForestPlusRegressor
    elif isinstance(fit, ClassifierMixin):
        RFPlus = RandomForestPlusClassifier
    else:
        raise ValueError("Unknown task.")
    rf_plus_model = RFPlus(rf_model=fit, **kwargs)
    rf_plus_model.fit(X, y)

    try:
        mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X, y=y, lfi=True, lfi_abs="outside")["lfi"].values
        if return_stability_scores:
            raise NotImplementedError
            stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
    except ValueError as e:
        if str(e) == 'Transformer representation was empty for all trees.':
            mdi_plus_scores = np.zeros((num_samples, num_features)) 
            stability_scores = None
        else:
            raise
    result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

    return result_table

def lime_local(X, y, fit):
    """
    Compute LIME local importance for each feature and sample.
    Larger values indicate more important features.
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

def tree_shap_local(X, y, fit):
    """
    Compute average treeshap value across observations.
    Larger absolute values indicate more important features.
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest (tree-based)
    :return: dataframe of shape: (n_samples, n_features)
    """
    explainer = shap.TreeExplainer(fit)
    shap_values = explainer.shap_values(X, check_additivity=False)
    if sklearn.base.is_classifier(fit):
        # Shape values are returned as a list of arrays, one for each class
        def add_abs(a, b):
            return abs(a) + abs(b)
        results = np.sum(np.abs(shap_values),axis=-1)
    else:
        results = abs(shap_values)
    result_table = pd.DataFrame(results, columns=[f'Feature_{i}' for i in range(X.shape[1])])

    return result_table

def permutation_local(X, y, fit, num_permutations=100):
    """
    Compute local permutation importance for each feature and sample.
    Larger values indicate more important features.
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


########## Use the following methods if evaluate on a separate test set

def LFI_test_evaluation_RF(X_train, y_train, X_test, y_test, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
    num_samples, num_features = X_test.shape
    if isinstance(fit, RegressorMixin):
        RFPlus = RandomForestPlusRegressor
    elif isinstance(fit, ClassifierMixin):
        RFPlus = RandomForestPlusClassifier
    else:
        raise ValueError("Unknown task.")
    rf_plus_model = RFPlus(rf_model=fit, **kwargs)
    rf_plus_model.fit(X_train, y_train)

    try:
        mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X_test, y=y_test, lfi=True, lfi_abs="none", sample_split=None, train_or_test = "test")["lfi"].values
        mdi_plus_scores = np.abs(mdi_plus_scores)
        if return_stability_scores:
            raise NotImplementedError
            stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
    except ValueError as e:
        if str(e) == 'Transformer representation was empty for all trees.':
            mdi_plus_scores = np.zeros((num_samples, num_features)) 
            stability_scores = None
        else:
            raise
    result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

    return result_table


def lime_test_evaluation_RF(X_train, y_train, X_test, y_test, fit):
    """
    Compute LIME local importance for each feature and sample.
    Larger values indicate more important features.
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest (tree-based)
    :return: dataframe of shape: (n_samples, n_features)

    """
    if isinstance(fit, RegressorMixin):
        mode='regression'
    elif isinstance(fit, ClassifierMixin):
        mode='classification'
    np.random.seed(1)
    num_samples, num_features = X_test.shape
    result = np.zeros((num_samples, num_features))
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, verbose=False, mode=mode)

    if mode == 'classification':
        if not hasattr(fit, 'predict_proba'):
            raise ValueError("Classifier model must have predict_proba method")

    for i in range(num_samples):
        if mode == 'classification':
            predict_fn = fit.predict_proba
        else:
            predict_fn = fit.predict
        exp = explainer.explain_instance(X_test[i], predict_fn, num_features=num_features)
        original_feature_importance = exp.as_map()[1]
        sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
        for j in range(num_features):
            result[i,j] = abs(sorted_feature_importance[j][1])
    # Convert the array to a DataFrame
    result_table = pd.DataFrame(result, columns=[f'Feature_{i}' for i in range(num_features)])

    return result_table


def tree_shap_evaluation_RF(X_train, y_train, X_test, y_test, fit):
    """
    Compute average treeshap value across observations.
    Larger absolute values indicate more important features.
    :param X: design matrix
    :param y: response
    :param fit: fitted model of interest (tree-based)
    :return: dataframe of shape: (n_samples, n_features)
    """
    explainer = shap.TreeExplainer(fit)
    shap_values = explainer.shap_values(X_test, check_additivity=False)
    if sklearn.base.is_classifier(fit):
        # Shape values are returned as a list of arrays, one for each class
        def add_abs(a, b):
            return abs(a) + abs(b)
        results = reduce(add_abs, shap_values)
    else:
        results = abs(shap_values)
    result_table = pd.DataFrame(results, columns=[f'Feature_{i}' for i in range(X_test.shape[1])])

    return result_table

def lime_test_evaluation_RF_plus(X_train, y_train, X_test, y_test, fit):
    num_samples, num_features = X_test.shape
    lime_scores = fit.get_lime_scores(X_train, X_test).values
    result_table = pd.DataFrame(lime_scores, columns=[f'Feature_{i}' for i in range(num_features)])

    return result_table


def kernel_shap_test_evaluation_RF_plus(X_train, y_train, X_test, y_test, fit):
    num_samples, num_features = X_test.shape
    kernel_shap_scores = fit.get_kernel_shap_scores(X_train, X_test)
    result_table = pd.DataFrame(kernel_shap_scores, columns=[f'Feature_{i}' for i in range(num_features)])

    return result_table


def LFI_test_evaluation_RF_plus(X_train, y_train, X_test, y_test, fit):
    num_samples, num_features = X_test.shape
    abs_lfi_scores = fit.get_mdi_plus_scores(X=X_test, y=y_test, lfi=True, lfi_abs="none", sample_split=None, train_or_test = "test")["lfi"].values
    abs_lfi_scores = np.abs(abs_lfi_scores)
    result_table = pd.DataFrame(abs_lfi_scores, columns=[f'Feature_{i}' for i in range(num_features)])

    return result_table


# def MDI_local_sub_stumps_evaluate(X_train, y_train, X_test, y_test, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
#     """
#     Compute local MDI importance for each feature and sample.
#     :param X: design matrix
#     :param y: response
#     :param fit: fitted model of interest (tree-based)
#     :return: dataframe of shape: (n_samples, n_features)

#     """
#     num_samples, num_features = X_test.shape
#     if isinstance(fit, RegressorMixin):
#         RFPlus = RandomForestPlusRegressor
#     elif isinstance(fit, ClassifierMixin):
#         RFPlus = RandomForestPlusClassifier
#     else:
#         raise ValueError("Unknown task.")
#     rf_plus_model = RFPlus(rf_model=fit, **kwargs)
#     rf_plus_model.fit(X_train, y_train)

#     try:
#         mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X_test, y=y_test, local_scoring_fns=scoring_fns, version = "sub", lfi=False, sample_split=None)["local"].values
#         if return_stability_scores:
#             raise NotImplementedError
#             stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
#     except ValueError as e:
#         if str(e) == 'Transformer representation was empty for all trees.':
#             mdi_plus_scores = np.zeros((num_samples, num_features)) 
#             stability_scores = None
#         else:
#             raise
#     result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

#     return result_table

# def lime_local(X_train, y_train, X_test, y_test, fit):
#     """
#     Compute LIME local importance for each feature and sample.
#     Larger values indicate more important features.
#     :param X: design matrix
#     :param y: response
#     :param fit: fitted model of interest (tree-based)
#     :return: dataframe of shape: (n_samples, n_features)

#     """
#     if isinstance(fit, RegressorMixin):
#         mode='regression'
#     elif isinstance(fit, ClassifierMixin):
#         mode='classification'
#     np.random.seed(1)
#     num_samples, num_features = X_test.shape
#     result = np.zeros((num_samples, num_features))
#     explainer = lime.lime_tabular.LimeTabularExplainer(X_train, verbose=False, mode=mode)
#     for i in range(num_samples):
#         exp = explainer.explain_instance(X_test[i], fit.predict, num_features=num_features)
#         original_feature_importance = exp.as_map()[1]
#         sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
#         for j in range(num_features):
#             result[i,j] = abs(sorted_feature_importance[j][1])
#     # Convert the array to a DataFrame
#     result_table = pd.DataFrame(result, columns=[f'Feature_{i}' for i in range(num_features)])

#     return result_table

# def MDI_local_all_stumps_evaluate(X_train, y_train, X_test, y_test, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
#     """
#     Wrapper around MDI+ object to get feature importance scores
    
#     :param X: ndarray of shape (n_samples, n_features)
#         The covariate matrix. If a pd.DataFrame object is supplied, then
#         the column names are used in the output
#     :param y: ndarray of shape (n_samples, n_targets)
#         The observed responses.
#     :param rf_model: scikit-learn random forest object or None
#         The RF model to be used for interpretation. If None, then a new
#         RandomForestRegressor or RandomForestClassifier is instantiated.
#     :param kwargs: additional arguments to pass to
#         RandomForestPlusRegressor or RandomForestPlusClassifier class.
#     :return: dataframe - [Var, Importance]
#                          Var: variable name
#                          Importance: MDI+ score
#     """
#     num_samples, num_features = X_test.shape
#     if isinstance(fit, RegressorMixin):
#         RFPlus = RandomForestPlusRegressor
#     elif isinstance(fit, ClassifierMixin):
#         RFPlus = RandomForestPlusClassifier
#     else:
#         raise ValueError("Unknown task.")
#     rf_plus_model = RFPlus(rf_model=fit, **kwargs)
#     rf_plus_model.fit(X_train, y_train)

#     try:
#         mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X_test, y=y_test, local_scoring_fns=scoring_fns, version = "all", lfi=False, sample_split=None)["local"].values
#         if return_stability_scores:
#             raise NotImplementedError
#             stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
#     except ValueError as e:
#         if str(e) == 'Transformer representation was empty for all trees.':
#             mdi_plus_scores = np.zeros((num_samples, num_features)) 
#             stability_scores = None
#         else:
#             raise
#     result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

#     return result_table

# def LFI_ablation_test_evaluation(X_train, y_train, X_test, y_test, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
#     num_samples, num_features = X_test.shape
#     if isinstance(fit, RegressorMixin):
#         RFPlus = RandomForestPlusRegressor
#     elif isinstance(fit, ClassifierMixin):
#         RFPlus = RandomForestPlusClassifier
#     else:
#         raise ValueError("Unknown task.")
#     rf_plus_model = RFPlus(rf_model=fit, **kwargs)
#     rf_plus_model.fit(X_train, y_train)

#     try:
#         mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X_test, y=y_test, lfi=True, lfi_abs="none", sample_split=None, train_or_test = "test")["lfi"].values
#         mdi_plus_scores = np.abs(mdi_plus_scores)
#         if return_stability_scores:
#             raise NotImplementedError
#             stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
#     except ValueError as e:
#         if str(e) == 'Transformer representation was empty for all trees.':
#             mdi_plus_scores = np.zeros((num_samples, num_features)) 
#             stability_scores = None
#         else:
#             raise
#     result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

#     return result_table

######################## Considering not using these methods
def LFI_sum_absolute_evaluate(X_train, y_train, X_test, y_test, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
    num_samples, num_features = X_test.shape
    if isinstance(fit, RegressorMixin):
        RFPlus = RandomForestPlusRegressor
    elif isinstance(fit, ClassifierMixin):
        RFPlus = RandomForestPlusClassifier
    else:
        raise ValueError("Unknown task.")
    rf_plus_model = RFPlus(rf_model=fit, **kwargs)
    rf_plus_model.fit(X_train, y_train)

    try:
        mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X_test, y=y_test, lfi=True, lfi_abs="inside", sample_split=None)["lfi"].values
        if return_stability_scores:
            raise NotImplementedError
            stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
    except ValueError as e:
        if str(e) == 'Transformer representation was empty for all trees.':
            mdi_plus_scores = np.zeros((num_samples, num_features)) 
            stability_scores = None
        else:
            raise
    result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

    return result_table

def LFI_sum_absolute(X, y, fit, scoring_fns="auto", return_stability_scores=False, **kwargs):
    num_samples, num_features = X.shape
    if isinstance(fit, RegressorMixin):
        RFPlus = RandomForestPlusRegressor
    elif isinstance(fit, ClassifierMixin):
        RFPlus = RandomForestPlusClassifier
    else:
        raise ValueError("Unknown task.")
    rf_plus_model = RFPlus(rf_model=fit, **kwargs)
    rf_plus_model.fit(X, y)

    try:
        mdi_plus_scores = rf_plus_model.get_mdi_plus_scores(X=X, y=y, lfi=True, lfi_abs="inside")["lfi"].values
        if return_stability_scores:
            raise NotImplementedError
            stability_scores = rf_plus_model.get_mdi_plus_stability_scores(B=25)
    except ValueError as e:
        if str(e) == 'Transformer representation was empty for all trees.':
            mdi_plus_scores = np.zeros((num_samples, num_features)) 
            stability_scores = None
        else:
            raise
    result_table = pd.DataFrame(mdi_plus_scores, columns=[f'Feature_{i}' for i in range(num_features)])

    return result_table