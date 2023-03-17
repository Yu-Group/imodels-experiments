import copy
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from feature_importance.util import ModelConfig
from imodels.importance.rf_plus import RandomForestPlusRegressor
from imodels.importance.ppms import RidgeRegressorPPM, LassoRegressorPPM


rf_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.33, random_state=42)
ridge_model = RidgeRegressorPPM()
lasso_model = LassoRegressorPPM()

ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33, 'random_state': 42})],
    [ModelConfig('RF-ridge', RandomForestPlusRegressor, model_type='tree',
                 other_params={'rf_model': copy.deepcopy(rf_model),
                               'prediction_model': copy.deepcopy(ridge_model)})],
    [ModelConfig('RF-lasso', RandomForestPlusRegressor, model_type='tree',
                 other_params={'rf_model': copy.deepcopy(rf_model),
                               'prediction_model': copy.deepcopy(lasso_model)})],
]

FI_ESTIMATORS = []
