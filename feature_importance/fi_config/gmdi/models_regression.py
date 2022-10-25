from sklearn.ensemble import RandomForestRegressor
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mda,lin_reg_marginal_t_test,tree_mdi, tree_mdi_OOB, tree_perm_importance, tree_shap,gjMDI
from imodels.importance.r2f_exp_cleaned import GMDI_pipeline, RidgePPM
from feature_importance.scripts.metrics import corr
import numpy as np
ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33, 'random_state': 27})]
]


FI_ESTIMATORS = [
    [FIModelConfig('GMDI_ridge_loo_r2', GMDI_pipeline, model_type='tree')],
    [FIModelConfig('GMDI_ridge_loo_r2_always_include_raw',GMDI_pipeline,model_type = 'tree', other_params = {'include_raw': False,'drop_features': False})],
    [FIModelConfig('GMDI_ridge_loo_r2_drop_raw',GMDI_pipeline,model_type = 'tree', other_params = {'include_raw': False})],
    [FIModelConfig('GMDI_ridge_loo_corr', GMDI_pipeline, model_type='tree', other_params = {'scoring_fn': corr})],
    [FIModelConfig('GMDI_ridge_loo_mdi_oob', GMDI_pipeline, model_type='tree', other_params = {'scoring_fn': 'mdi_oob'})],
    [FIModelConfig('GMDI_ridge_oob_r2', GMDI_pipeline, model_type='tree', other_params = {'oob': True, 'partial_prediction_model': RidgePPM(alphas=np.logspace(-4, 3, 100))})],
    [FIModelConfig('GMDI_ridge_oob_corr', GMDI_pipeline, model_type='tree', other_params = {'oob': True, 'partial_prediction_model': RidgePPM(alphas=np.logspace(-4, 3, 100)), 'scoring_fn': corr})],
    [FIModelConfig('GMDI_ridge_oob_mdi_oob', GMDI_pipeline, model_type='tree', other_params = {'oob': True, 'partial_prediction_model': RidgePPM(alphas=np.logspace(-4, 3, 100)), 'scoring_fn': 'mdi_oob'})],   
    #[FIModelConfig('GPermutation_ridge', GMDI_pipeline, model_type='tree', other_params = {'mode': 'keep_rest'})],
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')],
    [FIModelConfig('Permutation', tree_perm_importance, model_type='tree')],
    [FIModelConfig('MDA', tree_mda, model_type='tree')],
    [FIModelConfig('TreeSHAP', tree_shap, model_type='tree')]
]

#FI_ESTIMATORS = [
   # [FIModelConfig('GMDI_ridge_all_old', GMDI_pipeline, model_type='tree')],
   # [FIModelConfig('GMDI_ridge_oob_new', GMDI_pipeline, model_type='tree', other_params={'oob': True, 'partial_prediction_model': RidgePPM()})],
   # #[FIModelConfig('GMDI_ridge_oob_new_-5_5_100', GMDI_pipeline, model_type='tree', other_params={'oob': True, 'partial_prediction_model': RidgePPM(alphas = np.logspace(-5,5,100))})],
   # [FIModelConfig('GMDI_ridge_oob_new_MDIOOB_test', GMDI_pipeline, model_type='tree', other_params={'oob': True, 'partial_prediction_model': RidgePPM(alphas = [1e-8]),'include_raw':False})],
   # [FIModelConfig('GMDI_ridge_drop_raw', GMDI_pipeline, model_type = 'tree', other_params = {'include_raw': False})],
    #[FIModelConfig('GMDI_ridge_drop_raw_oob', GMDI_pipeline, model_type = 'tree', other_params = {'subsetting_scheme':'oob','include_raw': False})],
    #[FIModelConfig('GMDI_ridge_oob', GMDI_pipeline, model_type='tree', other_params = {'subsetting_scheme' : 'oob'})],
    #[FIModelConfig('GMDI_ridge_always_include_raw', GMDI_pipeline, model_type = 'tree', other_params = {'drop_features' : False})],
    #[FIModelConfig('GMDI_ridge_always_include_raw_oob', GMDI_pipeline, model_type = 'tree', other_params = {'drop_features' : False, 'subsetting_scheme' : 'oob'})],
    #[FIModelConfig('GPermutation_ridge', GMDI_pipeline, model_type='tree', other_params = {'mode': 'keep_rest'})],
    #[FIModelConfig('GPermutation_ridge_oob', GMDI_pipeline, model_type='tree', other_params = {'mode': 'keep_rest', 'subsetting_scheme': 'oob'})],
    #[FIModelConfig('GPermutation_ridge_always_include_raw', GMDI_pipeline, model_type='tree', other_params = {'mode': 'keep_rest','drop_features':False})],
    #[FIModelConfig('GPermutation_ridge_oob_always_include_raw', GMDI_pipeline, model_type='tree', other_params = {'mode': 'keep_rest', 'subsetting_scheme': 'oob','drop_features':False})],
    #[FIModelConfig('MDI', tree_mdi, model_type='tree')],
   # [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')],
    #[FIModelConfig('Permutation', tree_perm_importance, model_type='tree')],
    #[FIModelConfig('TreeSHAP', tree_shap, model_type='tree')],
    #[FIModelConfig('Ttest',lin_reg_marginal_t_test,ascending = False,model_type = 'linear')],
    #[FIModelConfig('MDA',tree_mda,model_type = 'tree')],                                                                                      
    #]
