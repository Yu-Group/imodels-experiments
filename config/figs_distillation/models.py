from functools import partial

import numpy as np
from numpy import concatenate as cat
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor

from imodels import FIGSRegressor, DistilledRegressor
from imodels import GreedyTreeRegressor, FIGSClassifier
from util import ModelConfig

RULEFIT_DEFAULT_KWARGS_CLASSIFICATION = {'random_state': 0, 'max_rules': None, 'include_linear': False, 'alpha': 1}
ESTIMATORS_CLASSIFICATION = [
    [ModelConfig('FIGS', FIGSClassifier, 'max_rules', n)
     for n in cat((np.arange(1, 19, 3), [25, 30]))],
    [ModelConfig('RandomForest', RandomForestClassifier)],  # single baseline
    [ModelConfig('GBDT-1', GradientBoostingClassifier, 'n_estimators', n, {'max_depth': 1})
     for n in cat((np.arange(1, 19, 3), [25, 30]))],
    [ModelConfig('GBDT-2', GradientBoostingClassifier, 'n_estimators', n, {'max_depth': 2})
     for n in np.arange(1, 9, 1)],
]

RULEFIT_DEFAULT_KWARGS_REGRESSION = {'random_state': 0, 'max_rules': None, 'include_linear': False, 'alpha': 0.15}
ESTIMATORS_REGRESSION = [
#     [ModelConfig('CART_(MSE)', GreedyTreeRegressor, 'max_depth', n)
#      for n in np.arange(1, 6, 1)],
#     [ModelConfig('FIGS', FIGSRegressor, 'max_rules', n)
#      for n in cat((np.arange(1, 19, 3), [25, 30]))],
#     [ModelConfig('RandomForest', RandomForestRegressor)],  # single baseline
#     [ModelConfig('GBDT', GradientBoostingRegressor)],  # single baseline
#     [ModelConfig('Dist-RF-FIGS', partial(DistilledRegressor,
#                                          teacher=RandomForestRegressor(),
#                                          student=FIGSRegressor(max_rules=n)),
#                  extra_aggregate_keys={'extra_max_rules': n})
#      for n in cat((np.arange(1, 19, 3), [25, 30]))],  # single baseline
#     [ModelConfig('Dist-GB-FIGS', partial(DistilledRegressor,
#                                          teacher=GradientBoostingRegressor(),
#                                          student=FIGSRegressor(max_rules=n)),
#                  extra_aggregate_keys={'extra_max_rules': n})              
#      for n in cat((np.arange(1, 19, 3), [25, 30]))],  # single baseline
    [ModelConfig('Dist-RF-FIGS-3', partial(DistilledRegressor,
                                         teacher=RandomForestRegressor(),
                                         student=FIGSRegressor(max_rules=n),
                                       n_iters_teacher=3),
                 extra_aggregate_keys={'extra_max_rules': n})
     for n in cat((np.arange(1, 19, 3), [25, 30]))],  # single baseline    

#     [ModelConfig('Dist-FIGS-FIGS', partial(DistilledRegressor,
#                                          teacher=FIGSRegressor(max_rules=30),
#                                          student=FIGSRegressor(max_rules=n)),
#                  extra_aggregate_keys={'extra_max_rules': n})              
#      for n in cat((np.arange(1, 19, 3), [25, 30]))],  # single baseline    


]

# python 01_fit_models.py --config figs_distillation --classification_or_regression regression --split_seed 0 --model Dist-GB-FIGS