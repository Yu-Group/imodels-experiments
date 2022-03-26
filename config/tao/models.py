from functools import partial

import numpy as np
from numpy import concatenate as cat

from imodels import GreedyTreeClassifier, FIGSClassifier, TaoTreeClassifier
from util import ModelConfig

# python 01_fit_models.py --config tao --classification_or_regression classification --model Tao --split_seed 0 --interactions_off
# python 01_fit_models.py --config tao --classification_or_regression classification --model Tao --split_seed 0 --ignore_cache --interactions_off

RULEFIT_DEFAULT_KWARGS_CLASSIFICATION = {'random_state': 0, 'max_rules': None, 'include_linear': False, 'alpha': 1}
ESTIMATORS_CLASSIFICATION = [
    [ModelConfig('FIGS', FIGSClassifier, 'max_rules', n)
     for n in cat((np.arange(1, 19, 3), [21]))],
    [ModelConfig('CART', GreedyTreeClassifier, 'max_depth', n)
     for n in np.arange(1, 6, 1)],
    [ModelConfig('TAO', partial(TaoTreeClassifier,
                                model_args={'max_leaf_nodes': n}),
                 extra_aggregate_keys={'max_leaf_nodes': n})
     for n in cat((np.arange(2, 19, 3), [21]))],
    # [ModelConfig('GBDT-1', GradientBoostingClassifier, 'n_estimators', n, {'max_depth': 1})
    #  for n in cat((np.arange(1, 19, 3), [25, 30]))],
    # [ModelConfig('GBDT-2', GradientBoostingClassifier, 'n_estimators', n, {'max_depth': 2})
    #  for n in np.arange(1, 9, 1)],
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
    #     [ModelConfig('FIGS_(Include_Linear)', FIGSRegressor, 'max_rules', n, {'include_linear': True})
    #      for n in cat(([30], np.arange(1, 19, 3), [25, 30]))],
    #     [ModelConfig('FIGS_(Reweighted)', FIGSRegressor, 'max_rules', n, {'posthoc_ridge': True})
    #      for n in cat((np.arange(1, 19, 3), [25, 30]))],
]
