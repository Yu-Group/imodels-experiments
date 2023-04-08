from functools import partial

from sklearn.ensemble import BaggingClassifier, BaggingRegressor, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from imodels.experimental.figs_ensembles import FIGSExtClassifier, FIGSExtRegressor
from imodels import FIGSClassifier
from util import ModelConfig
# from xgboost import XGBClassifier
from pygam import LogisticGAM, s

# python 01_fit_models.py --config figs_ensembles --classification_or_regression classification --model Boosting-FIGS
# python 01_fit_models.py --config figs_ensembles --classification_or_regression classification --model Bagging-FIGS-rerun
# python 01_fit_models.py --config figs_ensembles --classification_or_regression classification --model GAM

ESTIMATORS_CLASSIFICATION = [
    # [ModelConfig('CART', partial(GridSearchCV,
    #                              estimator=DecisionTreeClassifier(),
    #                              scoring='roc_auc',
    #                              param_grid={'max_depth': [1, 2, 3, 4, 5, 7]}))],
    # [ModelConfig('Linear', LogisticRegressionCV)],
    # [ModelConfig('RandomForest', RandomForestClassifier,
                 # other_params={'n_estimators': 100})],    
    # [ModelConfig('FIGS', partial(GridSearchCV,
                                 # estimator=FIGSExtClassifier(),
                                 # scoring='roc_auc',
                                 # param_grid={'max_rules': [1, 2**2, 2**3, 2**4, 2**5, 2**7]}))],
    # [ModelConfig('XGBoost', XGBClassifier)],
    # [ModelConfig('Bagging-FIGS', partial(BaggingClassifier,
                                   # base_estimator=FIGSExtClassifier(max_features=None, max_rules=10)),
                 # other_params={'n_estimators': 100})],   
    [ModelConfig('GAM', LogisticGAM)],  # defaults to univariate spline per feature, faster if scikit-sparse is installed
    # [ModelConfig('Boosting-FIGS', partial(AdaBoostClassifier,
    #                                base_estimator=FIGSExtClassifier(max_features=None, max_rules=10)),
    #              other_params={'n_estimators': 100})],     
    # [ModelConfig('Bagging-FIGS-sqrt-features', partial(BaggingClassifier,
    #                                base_estimator=FIGSClassifier(max_features='sqrt', max_rules=10)),
    #              other_params={'n_estimators': 100})],             
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
#     [ModelConfig('RFFIGS-10', partial(BaggingRegressor,
#                                    base_estimator=FIGSExtRegressor(max_features='auto', max_rules=10)),
#                  other_params={'n_estimators': 100})],          
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
#     [ModelConfig('RFFIGS-10-sub50', partial(BaggingClassifier,
#                                             max_samples=0.5,
#                                    base_estimator=FIGSExtClassifier(max_features='auto', max_rules=10)),
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
#     [ModelConfig('RFFIGS-10sub50', partial(BaggingRegressor,
#                                            max_samples=0.5,
#                                    base_estimator=FIGSExtRegressor(max_features='auto', max_rules=10)),
#                  other_params={'n_estimators': 100})],     