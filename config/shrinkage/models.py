from functools import partial

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from imodels import (
    GreedyTreeClassifier, GreedyTreeRegressor, HSTreeClassifierCV, HSTreeRegressorCV
)
from util import ModelConfig

RANDOM_FOREST_DEFAULT_KWARGS = {'random_state': 0}
ESTIMATORS_CLASSIFICATION = [
    [ModelConfig('CART', GreedyTreeClassifier, 'max_depth', n)
     for n in [1, 2, 3, 5, 7, 10]],
    [ModelConfig('ShrunkCART', partial(HSTreeClassifierCV, estimator=DecisionTreeClassifier(max_depth=n)),
                 'max_depth', n)
     for n in [1, 2, 3, 5, 7, 10]],
    [ModelConfig('Random_Forest', RandomForestClassifier, 'n_estimators', n, other_params=RANDOM_FOREST_DEFAULT_KWARGS)
     for n in [3, 10, 25, 50]],
    [ModelConfig('Gradient_Boosting', GradientBoostingClassifier, 'n_estimators', n,
                 other_params=RANDOM_FOREST_DEFAULT_KWARGS)
     for n in [10, 50, 100]],
]

ENSEMBLE_ESTIMATOR_NUMS = [3, 10, 25, 50]
TREE_DEPTHS = [1, 2, 3, 4, 5, 7, 8, 10, 15, 20, 25]
ESTIMATORS_REGRESSION = [
    [ModelConfig('CART_(MSE)', GreedyTreeRegressor, 'max_depth', n)
     for n in TREE_DEPTHS],
    # [Model('CART_(MAE)', GreedyTreeRegressor, 'max_depth', n, other_params={'criterion': 'absolute_error'})
    #  for n in TREE_DEPTHS],
    [ModelConfig('ShrunkCART', partial(HSTreeRegressorCV, estimator_=DecisionTreeRegressor(max_depth=n)))
     for n in TREE_DEPTHS],
    [ModelConfig('Random_Forest', RandomForestRegressor, other_params={'n_estimators': n})
     for n in ENSEMBLE_ESTIMATOR_NUMS],
    [ModelConfig('Shrunk_Random_Forest',
                 partial(HSTreeRegressorCV, estimator_=RandomForestRegressor(n_estimators=n)))
     for n in ENSEMBLE_ESTIMATOR_NUMS],
    [ModelConfig('Gradient_Boosting', GradientBoostingRegressor, 'n_estimators', n,
                 other_params=RANDOM_FOREST_DEFAULT_KWARGS)
     for n in ENSEMBLE_ESTIMATOR_NUMS],
    [ModelConfig('Shrunk_Gradient_Boosting',
                 partial(HSTreeRegressorCV, estimator_=GradientBoostingRegressor(n_estimators=n)))
     for n in ENSEMBLE_ESTIMATOR_NUMS],
]
