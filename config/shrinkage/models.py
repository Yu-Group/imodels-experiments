from functools import partial

from imodels import (
    GreedyTreeClassifier, GreedyTreeRegressor, ShrunkTreeClassifierCV, ShrunkTreeRegressorCV
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from util import Model

RANDOM_FOREST_DEFAULT_KWARGS = {'random_state': 0}
ESTIMATORS_CLASSIFICATION = [
    [Model('CART', GreedyTreeClassifier, 'max_depth', n)
     for n in [1, 2, 3, 5, 7, 10]],
    [Model('ShrunkCART', partial(ShrunkTreeClassifierCV, estimator=DecisionTreeClassifier(max_depth=n)), 'max_depth', n)
     for n in [1, 2, 3, 5, 7, 10]],
    [Model('Random_Forest', RandomForestClassifier, 'n_estimators', n, other_params=RANDOM_FOREST_DEFAULT_KWARGS)
     for n in [3, 10, 25, 50]],
    [Model('Gradient_Boosting', GradientBoostingClassifier, 'n_estimators', n,
           other_params=RANDOM_FOREST_DEFAULT_KWARGS)
     for n in [10, 50, 100]],
]

ENSEMBLE_ESTIMATOR_NUMS = [3, 10, 25, 50]
TREE_DEPTHS = [1, 2, 3, 4, 5, 7, 8, 10, 15, 20, 25]
ESTIMATORS_REGRESSION = [
    [Model('CART_(MSE)', GreedyTreeRegressor, 'max_depth', n)
     for n in TREE_DEPTHS],
    # [Model('CART_(MAE)', GreedyTreeRegressor, 'max_depth', n, other_params={'criterion': 'absolute_error'})
    #  for n in TREE_DEPTHS],
    [Model('ShrunkCART', partial(ShrunkTreeRegressorCV, estimator_=DecisionTreeRegressor(max_depth=n)))
     for n in TREE_DEPTHS],
    [Model('Random_Forest', RandomForestRegressor, other_params={'n_estimators': n})
     for n in ENSEMBLE_ESTIMATOR_NUMS],
    [Model('Shrunk_Random_Forest',
           partial(ShrunkTreeRegressorCV, estimator_=RandomForestRegressor(n_estimators=n)))
     for n in ENSEMBLE_ESTIMATOR_NUMS],
    [Model('Gradient_Boosting', GradientBoostingRegressor, 'n_estimators', n, other_params=RANDOM_FOREST_DEFAULT_KWARGS)
     for n in ENSEMBLE_ESTIMATOR_NUMS],
    [Model('Shrunk_Gradient_Boosting',
           partial(ShrunkTreeRegressorCV, estimator_=GradientBoostingRegressor(n_estimators=n)))
     for n in ENSEMBLE_ESTIMATOR_NUMS],
]
