import numpy as np
from numpy import concatenate as cat
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from imodels import GreedyTreeClassifier, GreedyTreeRegressor, C45TreeClassifier, SaplingSumClassifier
from imodels import RuleFitClassifier, RuleFitRegressor, SaplingSumRegressor
from util import ModelConfig

RULEFIT_DEFAULT_KWARGS_CLASSIFICATION = {'random_state': 0, 'max_rules': None, 'include_linear': False, 'alpha': 1}
ESTIMATORS_CLASSIFICATION = [
#     [ModelConfig('SAPS', SaplingSumClassifier, 'max_rules', n)
#      for n in cat((np.arange(1, 19, 3), [25, 30]))],
#     [ModelConfig('C45', C45TreeClassifier, 'max_rules', n)
#      for n in np.concatenate((np.arange(1, 19, 3), [25, 30]))],
#     [ModelConfig('CART', GreedyTreeClassifier, 'max_depth', n)
#      for n in np.arange(1, 6, 1)],
#     [ModelConfig('Rulefit', RuleFitClassifier, 'n_estimators', n, RULEFIT_DEFAULT_KWARGS_CLASSIFICATION)
#      for n in np.arange(1, 11, 1)],  # can also vary n_estimators and get a good spread
#     [ModelConfig('RandomForest', RandomForestClassifier)],  # single baseline
    [ModelConfig('GBDT-1', GradientBoostingClassifier, 'n_estimators', n, {'max_depth': 1})
     for n in cat((np.arange(1, 19, 3), [25, 30]))],  
    [ModelConfig('GBDT-2', GradientBoostingClassifier, 'n_estimators', n, {'max_depth': 2})
     for n in np.arange(1, 9, 1)],      
#     [ModelConfig('SAPS_(Include_Linear)', SaplingSumClassifier, 'max_rules', n, {'include_linear': True})
#      for n in cat((np.arange(1, 19, 3), [25, 30]))],
#     [ModelConfig('SAPS_(Reweighted)', SaplingSumClassifier, 'max_rules', n, {'posthoc_ridge': True})
#      for n in cat((np.arange(1, 19, 3), [25, 30]))],    
]

RULEFIT_DEFAULT_KWARGS_REGRESSION = {'random_state': 0, 'max_rules': None, 'include_linear': False, 'alpha': 0.15}
ESTIMATORS_REGRESSION = [
#     [ModelConfig('Rulefit', RuleFitRegressor, 'n_estimators', n, RULEFIT_DEFAULT_KWARGS_REGRESSION)
#      for n in cat((np.arange(1, 11, 1), [15]))],  # can also vary n_estimators and get a good spread
#     [ModelConfig('CART_(MSE)', GreedyTreeRegressor, 'max_depth', n)
#      for n in np.arange(1, 6, 1)],
#     [ModelConfig('CART_(MAE)', GreedyTreeRegressor, 'max_depth', n, {'criterion': 'absolute_error'})
#      for n in np.arange(1, 6, 1)],
#     [ModelConfig('SAPS', SaplingSumRegressor, 'max_rules', n)
#      for n in cat((np.arange(1, 19, 3), [25, 30]))],    
#     [ModelConfig('RandomForest', RandomForestRegressor)],  # single baseline    
    [ModelConfig('GBDT-1', GradientBoostingRegressor, 'n_estimators', n, {'max_depth': 1})
     for n in cat((np.arange(1, 19, 3), [25, 30]))],  
    [ModelConfig('GBDT-2', GradientBoostingRegressor, 'n_estimators', n, {'max_depth': 2})
     for n in np.arange(1, 9, 1)],          
#     [ModelConfig('SAPS_(Include_Linear)', SaplingSumRegressor, 'max_rules', n, {'include_linear': True})
#      for n in cat(([30], np.arange(1, 19, 3), [25, 30]))],
#     [ModelConfig('SAPS_(Reweighted)', SaplingSumRegressor, 'max_rules', n, {'posthoc_ridge': True})
#      for n in cat((np.arange(1, 19, 3), [25, 30]))],
]
