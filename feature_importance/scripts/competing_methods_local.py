import os
import sys
import pandas as pd
import numpy as np
import sklearn.base
from sklearn.base import RegressorMixin, ClassifierMixin
from functools import reduce

import shap


def tree_shap_local(X, y, fit):
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
    shap_values = explainer.shap_values(X, check_additivity=False)
    if sklearn.base.is_classifier(fit):
        def add_abs(a, b):
            return abs(a) + abs(b)
        results = reduce(add_abs, shap_values)
    else:
        results = abs(shap_values)
    result_table = pd.DataFrame(results)
    # results = results.mean(axis=0)
    # results = pd.DataFrame(data=results, columns=['importance'])
    # # Use column names from dataframe if possible
    # if isinstance(X, pd.DataFrame):
    #     results.index = X.columns
    # results.index.name = 'var'
    # results.reset_index(inplace=True)

    return result_table