from sklearn.ensemble import RandomForestRegressor
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import lin_reg_marginal_t_test,tree_mdi, tree_mdi_OOB, tree_perm_importance, tree_shap
from imodels.importance.r2f_exp_cleaned import GMDI_pipeline


ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33, 'random_state': 42})]
]

FI_ESTIMATORS = [
    [FIModelConfig('GMDI_ridge', GMDI_pipeline, model_type='tree')],
    [FIModelConfig('GMDI_ridge_oob', GMDI_pipeline, model_type='tree', other_params = {'subsetting_scheme': 'oob'})],
    [FIModelConfig('GPermutation_ridge', GMDI_pipeline, model_type='tree', other_params = {'mode': 'keep_rest'})],
    [FIModelConfig('GPermutation_ridge_oob', GMDI_pipeline, model_type='tree', other_params = {'mode': 'keep_rest', 'subsetting_scheme': 'oob'})],
    [FIModelConfig('MDI', tree_mdi, model_type='tree')],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='tree')],
    [FIModelConfig('Permutation', tree_perm_importance, model_type='tree')],
    [FIModelConfig('TreeSHAP', tree_shap, model_type='tree')],
    [FIModelConfig('Ttest',lin_reg_marginal_t_test,ascending = False,model_type = 'linear')],
    [FIModelConfig('MDA',tree_mda,model_type = 'tree')],                                                                                      
]
