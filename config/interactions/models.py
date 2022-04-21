from functools import partial

import numpy as np
from numpy import concatenate as cat
from sklearn.tree import DecisionTreeClassifier

from imodels import GreedyTreeClassifier, FIGSClassifier, TaoTreeClassifier, RuleFitClassifier, HSTreeClassifierCV
# from imodels.experimental.interactions import InteractionsClassifier
from util import ModelConfig

# example running all classification models on all datasets
# python 01_fit_models.py --config interactions --classification_or_regression classification

# example running a single model with a different seed
# python 01_fit_models.py --config interactions --classification_or_regression classification --model FIGS --split_seed 1

# example running ignoring cache
# python 01_fit_models.py --config interactions --classification_or_regression classification --ignore_cache

RULEFIT_DEFAULT_KWARGS_CLASSIFICATION = {'random_state': 0, 'max_rules': None, 'include_linear': False, 'alpha': 1}
ESTIMATORS_CLASSIFICATION = [
    [ModelConfig('InteractionsClassifier', FIGSClassifier, 'max_rules', n)
     for n in cat((np.arange(1, 19, 3), [21]))],
    [ModelConfig('HSCART', partial(HSTreeClassifierCV, estimator_=DecisionTreeClassifier(max_leaf_nodes=n + 1)),
                 'max_depth', n)
     for n in cat((np.arange(1, 19, 3), [21]))],
    [ModelConfig('FIGS', FIGSClassifier, 'max_rules', n)
     for n in cat((np.arange(1, 19, 3), [21]))],
    [ModelConfig('CART', GreedyTreeClassifier, 'max_depth', n)
     for n in np.arange(1, 6, 1)],
    [ModelConfig('Rulefit', RuleFitClassifier, 'n_estimators', n, RULEFIT_DEFAULT_KWARGS_CLASSIFICATION)
     for n in np.arange(1, 11, 1)],  # can also vary n_estimators and get a good spread
]

RULEFIT_DEFAULT_KWARGS_REGRESSION = {'random_state': 0, 'max_rules': None, 'include_linear': False, 'alpha': 0.15}
ESTIMATORS_REGRESSION = [
    #     [ModelConfig('Rulefit', RuleFitRegressor, 'n_estimators', n, RULEFIT_DEFAULT_KWARGS_REGRESSION)
    #      for n in cat((np.arange(1, 11, 1), [15]))],  # can also vary n_estimators and get a good spread
    #     [ModelConfig('CART_(MSE)', GreedyTreeRegressor, 'max_depth', n)
    #      for n in np.arange(1, 6, 1)],
    #     [ModelConfig('CART_(MAE)', GreedyTreeRegressor, 'max_depth', n, {'criterion': 'absolute_error'})
    #      for n in np.arange(1, 6, 1)],
    #     [ModelConfig('FIGS', FIGSRegressor, 'max_rules', n)
    #      for n in cat((np.arange(1, 19, 3), [25, 30]))],
    #     [ModelConfig('RandomForest', RandomForestRegressor)],  # single baseline
    #     [ModelConfig('GBDT-1', GradientBoostingRegressor, 'n_estimators', n, {'max_depth': 1})
    #      for n in cat((np.arange(1, 19, 3), [25, 30]))],
    #     [ModelConfig('GBDT-2', GradientBoostingRegressor, 'n_estimators', n, {'max_depth': 2})
    #      for n in np.arange(1, 9, 1)],
]
