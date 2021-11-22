import numpy as np
from imodels import (
    GreedyTreeClassifier, GreedyTreeRegressor, ShrunkTreeCV,
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)

from util import Model

RANDOM_FOREST_DEFAULT_KWARGS = {'random_state': 0}
ESTIMATORS_CLASSIFICATION = (
    [Model('CART', GreedyTreeClassifier, 'max_depth', n)
     for n in [1, 2, 3, 5, 7, 10]],
    [Model('Random_Forest', RandomForestClassifier, 'n_estimators', n, other_params=RANDOM_FOREST_DEFAULT_KWARGS)
     for n in [10, 50, 100]],
    [Model('Gradient_Boosting', GradientBoostingClassifier, 'n_estimators', n,
           other_params=RANDOM_FOREST_DEFAULT_KWARGS)
     for n in [10, 50, 100]],
)
l = [GreedyTreeClassifier, GreedyTreeRegressor, ShrunkTreeCV, ]

ESTIMATORS_REGRESSION = (
    [Model('CART_(MSE)', GreedyTreeRegressor, 'max_depth', n)
     for n in [1, 2, 3, 5, 7, 10]],
    [Model('CART_(MAE)', GreedyTreeRegressor, 'max_depth', n, other_params={'criterion': 'absolute_error'})
     for n in [1, 2, 3, 5, 7, 10]],
    [Model('Random_Forest', RandomForestRegressor, 'n_estimators', n, other_params=RANDOM_FOREST_DEFAULT_KWARGS)
     for n in [10, 50, 100]],
    [Model('Gradient_Boosting', GradientBoostingRegressor, 'n_estimators', n, other_params=RANDOM_FOREST_DEFAULT_KWARGS)
     for n in [10, 50, 100]],
)
