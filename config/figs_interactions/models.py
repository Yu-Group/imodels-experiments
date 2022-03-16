from functools import partial

import numpy as np
# from bartpy.initializers.sklearntreeinitializer import SklearnTreeInitializer
# from imodels.tree.iterative_random_forest.iterative_random_forest import IRFClassifier
from imodels.experimental.bartpy.initializers.sklearntreeinitializer import SklearnTreeInitializer
from imodels.tree.figs import FIGSClassifierCV, FIGSRegressorCV
from numpy import concatenate as cat
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor

from imodels import FIGSRegressor, DistilledRegressor, BART
from imodels import GreedyTreeRegressor, FIGSClassifier
# from irf.ensemble import RandomForestRegressorWithWeights
# from bartpy import BART
from sklearn.model_selection import GridSearchCV

from util import ModelConfig

RULEFIT_DEFAULT_KWARGS_CLASSIFICATION = {'random_state': 0, 'max_rules': None, 'include_linear': False, 'alpha': 1}
ESTIMATORS_CLASSIFICATION = [
    [ModelConfig('FIGS', FIGSClassifierCV)],
    [ModelConfig('BFIGS',
                 partial(BART, classification=True, initializer=SklearnTreeInitializer(tree_=FIGSClassifierCV())))],
    [ModelConfig('RandomForest', RandomForestClassifier)],  # single baseline
    [ModelConfig('GBDT-1', GradientBoostingClassifier)]
]

RULEFIT_DEFAULT_KWARGS_REGRESSION = {'random_state': 0, 'max_rules': None, 'include_linear': False, 'alpha': 0.15}
ESTIMATORS_REGRESSION = [
    [ModelConfig('FIGS', FIGSRegressorCV)],
    [ModelConfig('BFIGS', partial(BART, initializer=SklearnTreeInitializer(tree_=FIGSRegressorCV())))],
    [ModelConfig('RandomForest', RandomForestRegressor)],  # single baseline
    [ModelConfig('GBDT-1', GradientBoostingRegressor, 'n_estimators', n, {'max_depth': 1})
     for n in cat((np.arange(3, 19, 3), [25, 30]))],
    [ModelConfig('GBDT-2', GradientBoostingRegressor, 'n_estimators', n, {'max_depth': 2})
     for n in np.arange(3, 9, 1)],
    # [ModelConfig('iRF',  RandomForestRegressorWithWeights, 'n_estimators', n)
    #  for n in cat((np.arange(3, 19, 3), [25, 30]))]
]

# python 01_fit_models.py --config figs_distillation --classification_or_regression regression --split_seed 0 --model Dist-GB-FIGS
