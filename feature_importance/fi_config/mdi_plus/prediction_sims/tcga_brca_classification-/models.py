import copy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from sklearn.utils.extmath import softmax
from feature_importance.util import ModelConfig
from imodels.importance.rf_plus import RandomForestPlusClassifier
from imodels.importance.ppms import RidgeClassifierPPM, LogisticClassifierPPM
      
      
rf_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, max_features='sqrt', random_state=42)
ridge_model = RidgeClassifierPPM()
logistic_model = LogisticClassifierPPM()

ESTIMATORS = [
    [ModelConfig('RF', RandomForestClassifier, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'random_state': 42})],
    [ModelConfig('RF-ridge', RandomForestPlusClassifier, model_type='tree',
                 other_params={'rf_model': copy.deepcopy(rf_model),
                               'prediction_model': copy.deepcopy(ridge_model)})],
    [ModelConfig('RF-logistic', RandomForestPlusClassifier, model_type='tree',
                 other_params={'rf_model': copy.deepcopy(rf_model),
                               'prediction_model': copy.deepcopy(logistic_model)})],
]

FI_ESTIMATORS = []
