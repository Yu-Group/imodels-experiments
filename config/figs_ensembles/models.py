from functools import partial

from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from imodels.experimental.figs_ensembles import FIGSExtClassifier, FIGSExtRegressor
from util import ModelConfig

ESTIMATORS_CLASSIFICATION = [
#     [ModelConfig('CART', partial(GridSearchCV,
#                                  estimator=DecisionTreeClassifier(),
#                                  scoring='roc_auc',
#                                  param_grid={'max_depth': [1, 3, 5, 7, 9]}))],
#     [ModelConfig('Linear', LogisticRegressionCV)],
#     [ModelConfig('RandomForest', RandomForestClassifier, other_params={'n_estimators': 100})],    
#     [ModelConfig('FIGS', partial(GridSearchCV,
#                                  estimator=FIGSExtClassifier(),
#                                  scoring='roc_auc',
#                                  param_grid={'max_rules': [1, 2**3, 2**5, 2**7, 2**9]}))],
#     [ModelConfig('BaggingFIGS', partial(BaggingClassifier, base_estimator=FIGSExtClassifier()),
#                  other_params={'n_estimators': 100})],
    [ModelConfig('RFFIGS', partial(BaggingClassifier,
                                   max_samples=1, # 1 is same as normal RF....
                                   base_estimator=FIGSExtClassifier(max_features='auto')),
                 other_params={'n_estimators': 10})],    
]

ESTIMATORS_REGRESSION = [
#     [ModelConfig('CART', partial(GridSearchCV,
#                                  estimator=DecisionTreeRegressor(),
#                                  scoring='roc_auc',
#                                  param_grid={'max_depth': [1, 3, 5, 7, 9]}))],
#     [ModelConfig('Linear', RidgeCV)],
#     [ModelConfig('RandomForest', RandomForestRegressor, other_params={'n_estimators': 100})],    
#     [ModelConfig('FIGS', partial(GridSearchCV,
#                                  estimator=FIGSExtRegressor(),
#                                  scoring='roc_auc',
#                                  param_grid={'max_rules': [1, 2**3, 2**5, 2**7, 2**9]}))],
#     [ModelConfig('BaggingFIGS', partial(BaggingRegressor, base_estimator=FIGSExtRegressor()),
#                  other_params={'n_estimators': 100})],
#     [ModelConfig('RFFIGS', partial(BaggingRegressor,
#                                    base_estimator=FIGSExtRegressor(max_features='auto')),
#                  other_params={'n_estimators': 100})],        
]
