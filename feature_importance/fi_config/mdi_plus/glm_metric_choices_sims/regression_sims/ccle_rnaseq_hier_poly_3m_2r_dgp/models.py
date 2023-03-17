from sklearn.ensemble import RandomForestRegressor
from feature_importance.util import ModelConfig, FIModelConfig, neg_mean_absolute_error
from feature_importance.scripts.competing_methods import tree_mdi_plus, tree_mdi, tree_mdi_OOB, tree_mda, tree_shap, tree_mdi_plus_ensemble
from imodels.importance.ppms import RidgeRegressorPPM, LassoRegressorPPM
from sklearn.metrics import mean_absolute_error, r2_score

ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33, 'random_state': 42})]
]

FI_ESTIMATORS = [
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('MDI+_ridge_r2', tree_mdi_plus, model_type='tree', other_params={'prediction_model': RidgeRegressorPPM(gcv_mode='eigen'), 'return_stability_scores': True})],
    [FIModelConfig('MDI+_lasso_r2', tree_mdi_plus, model_type='tree', other_params={'prediction_model': LassoRegressorPPM(), 'return_stability_scores': True})],
    [FIModelConfig('MDI+_ridge_neg_mae', tree_mdi_plus, model_type='tree', other_params={'prediction_model': RidgeRegressorPPM(gcv_mode='eigen'), 'scoring_fns': neg_mean_absolute_error, 'return_stability_scores': True})],
    [FIModelConfig('MDI+_lasso_neg_mae', tree_mdi_plus, model_type='tree', other_params={'prediction_model': LassoRegressorPPM(), 'scoring_fns': neg_mean_absolute_error, 'return_stability_scores': True})],
    [FIModelConfig('MDI+_ensemble', tree_mdi_plus_ensemble, model_type='tree', ascending=False,
                   other_params={"ridge": {"prediction_model": RidgeRegressorPPM(gcv_mode='eigen')},
                                 "lasso": {"prediction_model": LassoRegressorPPM()},
                                 "scoring_fns": {"r2": r2_score, "mae": neg_mean_absolute_error}})],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')]
]
