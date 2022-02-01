import numpy as np
from numpy import concatenate as cat

from imodels import GreedyTreeClassifier, GreedyTreeRegressor, C45TreeClassifier, FIGSClassifier, HSTreeClassifier, \
    HSTreeRegressor
from imodels import RuleFitClassifier, RuleFitRegressor, FIGSRegressor
from util import ModelConfig

RULEFIT_DEFAULT_KWARGS_CLASSIFICATION = {'random_state': 0, 'max_rules': None, 'include_linear': False, 'alpha': 1}
ESTIMATORS_CLASSIFICATION = [
    [ModelConfig('FIGS', FIGSClassifier, 'max_rules', n)
     for n in cat((np.arange(1, 19, 3), [25, 30]))],
    # [ModelConfig('SAPS_(Include_Linear)', FIGSClassifier, 'max_rules', n, {'include_linear': True})
     # for n in cat((np.arange(1, 19, 3), [25, 30]))],
    # [ModelConfig('SAPS_(Reweighted)', FIGSClassifier, 'max_rules', n, {'posthoc_ridge': True})
     # for n in cat((np.arange(1, 19, 3), [25, 30]))],
    [ModelConfig('C45', C45TreeClassifier, 'max_rules', n)
     for n in np.concatenate((np.arange(1, 19, 3), [25, 30]))],
    [ModelConfig('CART', GreedyTreeClassifier, 'max_depth', n)
     for n in np.arange(1, 6, 3)],
    [ModelConfig('hsCART', HSTreeClassifier, 'max_depth', n)
     for n in np.arange(1, 6, 3)],
    [ModelConfig('Rulefit', RuleFitClassifier, 'n_estimators', n, RULEFIT_DEFAULT_KWARGS_CLASSIFICATION)
     for n in [1, 3]],  # can also vary n_estimators and get a good spread
]

RULEFIT_DEFAULT_KWARGS_REGRESSION = {'random_state': 0, 'max_rules': None, 'include_linear': False, 'alpha': 0.15}
ESTIMATORS_REGRESSION = [
    # [ModelConfig('SAPS_(Include_Linear)', FIGSRegressor, 'max_rules', n, {'include_linear': True})
    #  for n in cat(([30], np.arange(1, 19, 3), [25, 30]))],
    # [ModelConfig('SAPS_(Reweighted)', FIGSRegressor, 'max_rules', n, {'posthoc_ridge': True})
    #  for n in cat((np.arange(1, 19, 3), [25, 30]))],
    [ModelConfig('Rulefit', RuleFitRegressor, 'n_estimators', n, RULEFIT_DEFAULT_KWARGS_REGRESSION)
     for n in [1, 3]],  # can also vary n_estimators and get a good spread
    [ModelConfig('CART_(MSE)', GreedyTreeRegressor, 'max_depth', n)
     for n in np.arange(1, 6, 3)],
    [ModelConfig('CART_(MAE)', GreedyTreeRegressor, 'max_depth', n, {'criterion': 'absolute_error'})
     for n in np.arange(1, 6, 3)],
    [ModelConfig('hsCART', HSTreeRegressor, 'max_depth', n)
     for n in np.arange(1, 6, 3)],
    [ModelConfig('FIGS', FIGSRegressor, 'max_rules', n)
     for n in cat((np.arange(1, 19, 3), [25, 30]))],
]
