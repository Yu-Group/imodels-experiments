from functools import partial

from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from imodels.experimental.figs_ensembles import FIGSExtClassifier, FIGSExtRegressor
from util import ModelConfig

ESTIMATORS_CLASSIFICATION = [
    [ModelConfig('CART', partial(GridSearchCV,
                                 estimator=DecisionTreeClassifier(),
                                 param_grid={'max_depth': [1, 3, 5, 7, 9]}))],
    [ModelConfig('Linear', LogisticRegressionCV)],
    [ModelConfig('FIGS', partial(GridSearchCV,
                                 estimator=FIGSExtClassifier(),
                                 param_grid={'max_rules': [1, 2**3, 2**5, 2**7, 2**9]}))],
    [ModelConfig('RandomForest', RandomForestClassifier, other_params={'n_estimators': 100})],
    [ModelConfig('BaggingFIGS', partial(BaggingClassifier, base_estimator=FIGSExtClassifier()),
                 other_params={'n_estimators': 100})],
]

ESTIMATORS_REGRESSION = [
    [ModelConfig('CART', partial(GridSearchCV,
                                 estimator=DecisionTreeRegressor(),
                                 param_grid={'max_depth': [1, 3, 5, 7, 9]}))],
    [ModelConfig('Linear', RidgeCV)],
    [ModelConfig('FIGS', partial(GridSearchCV,
                                 estimator=FIGSExtRegressor(),
                                 param_grid={'max_rules': [1, 2**3, 2**5, 2**7, 2**9]}))],
    [ModelConfig('RandomForest', RandomForestRegressor, other_params={'n_estimators': 100})],
    [ModelConfig('BaggingFIGS', partial(BaggingRegressor, base_estimator=FIGSExtRegressor()),
                 other_params={'n_estimators': 100})],
]
