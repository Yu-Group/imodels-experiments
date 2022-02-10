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
#                                  param_grid={'max_depth': [1, 2, 3, 4, 5, 7]}))],
#     [ModelConfig('Linear', LogisticRegressionCV)],
#     [ModelConfig('RandomForest', RandomForestClassifier,
#                  other_params={'n_estimators': 100})],    
#     [ModelConfig('FIGS', partial(GridSearchCV,
#                                  estimator=FIGSExtClassifier(),
#                                  scoring='roc_auc',
#                                  param_grid={'max_rules': [1, 2**2, 2**3, 2**4, 2**5, 2**7]}))],
    [ModelConfig('RFFIGS-10', partial(BaggingClassifier,
                                   base_estimator=FIGSExtClassifier(max_features='auto', max_rules=10)),
                 other_params={'n_estimators': 100})],         
] 

ESTIMATORS_REGRESSION = [
#     [ModelConfig('CART', partial(GridSearchCV,
#                                  estimator=DecisionTreeRegressor(),
#                                  scoring='r2',
#                                  param_grid={'max_depth': [1, 2, 3, 4, 5, 7]}))],
#     [ModelConfig('Linear', RidgeCV)],
#     [ModelConfig('RandomForest', RandomForestRegressor, other_params={'n_estimators': 100})],    
#     [ModelConfig('FIGS', partial(GridSearchCV,
#                                  estimator=FIGSExtRegressor(),
#                                  scoring='r2',
#                                  param_grid={'max_rules': [1, 2**2, 2**3, 2**4, 2**5, 2**7]}))],
    [ModelConfig('RFFIGS-10', partial(BaggingRegressor,
                                   base_estimator=FIGSExtRegressor(max_features='auto', max_rules=10)),
                 other_params={'n_estimators': 100})],         
]



################# Old Configurations Classification


#     [ModelConfig('RFFIGS', partial(BaggingClassifier,
#                                    base_estimator=FIGSExtClassifier(max_features='auto')),
#                  other_params={'n_estimators': 100})],   
#     [ModelConfig('BaggingFIGS', partial(BaggingClassifier, base_estimator=FIGSExtClassifier()),
#                  other_params={'n_estimators': 100})], 
#     [ModelConfig('RFFIGS-log2', partial(BaggingClassifier,
#                                    base_estimator=FIGSExtClassifier(max_features='log2')),
#                  other_params={'n_estimators': 100})],    
#     [ModelConfig('RFFIGS-depth3', partial(BaggingClassifier,
#                                    base_estimator=FIGSExtClassifier(max_features='auto', max_rules=2**3)),
#                  other_params={'n_estimators': 100})],      
#     [ModelConfig('RFFIGS-depth4', partial(BaggingClassifier,
#                                    base_estimator=FIGSExtClassifier(max_features='auto', max_rules=2**4)),
#                  other_params={'n_estimators': 100})],   


################# Old Configurations Regression

#     [ModelConfig('BaggingFIGS', partial(BaggingRegressor, base_estimator=FIGSExtRegressor()),
#                  other_params={'n_estimators': 100})],    
#     [ModelConfig('RFFIGS-log2', partial(BaggingRegressor,
#                                    base_estimator=FIGSExtRegressor(max_features='log2')),
#                  other_params={'n_estimators': 100})],       
#     [ModelConfig('RFFIGS', partial(BaggingRegressor,
#                                    base_estimator=FIGSExtRegressor(max_features='auto')),
#                  other_params={'n_estimators': 100})],       

#     [ModelConfig('RFFIGS-depth3', partial(BaggingRegressor,
#                                    base_estimator=FIGSExtRegressor(max_features='auto', max_rules=2**3)),
#                  other_params={'n_estimators': 100})],      
#     [ModelConfig('RFFIGS-depth4', partial(BaggingRegressor,
#                                    base_estimator=FIGSExtRegressor(max_features='auto', max_rules=2**4)),
#                  other_params={'n_estimators': 100})],    