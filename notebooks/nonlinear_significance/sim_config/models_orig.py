# Note: This is currently just a test file with dummy/placeholder functions

from functools import partial

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from imodels import (
    GreedyTreeClassifier, GreedyTreeRegressor, HSTreeClassifierCV, HSTreeRegressorCV
)
from util import ModelConfig, FIModelConfig

from nonlinear_significance.scripts.methods import lin_reg_t_test, tree_mdi, perm_importance,tree_shap_mean, tree_feature_significance, optimal_tree_feature_significance, tree_mdi_OOB, lin_reg_marginal_t_test
#knockpy_swap_integral

ESTIMATORS = [
    # [ModelConfig('RF50', RandomForestRegressor,other_params = {'n_estimators':50, 'min_samples_leaf':5, 'max_features':0.33}, model_type='tree')],
    # [ModelConfig('RF100', RandomForestRegressor,other_params = {'n_estimators':100, 'min_samples_leaf':5, 'max_features':0.33}, model_type='tree')],
    # [ModelConfig('RF500', RandomForestRegressor,other_params = {'n_estimators':500, 'min_samples_leaf':5, 'max_features':0.33}, model_type='tree')],
    # [ModelConfig('RF1000', RandomForestRegressor,other_params = {'n_estimators':1000, 'min_samples_leaf':5, 'max_features':0.33}, model_type='tree')],
    [ModelConfig('RF', RandomForestRegressor, other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33}, model_type='tree')],
    # [ModelConfig('OLS', LinearRegression, model_type='linear')],
]

FI_ESTIMATORS = [
    [FIModelConfig('r2f', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'lasso', 'max_components_type': 'minfracnp', 'fraction_chosen': 0.5, 'criteria': 'bic', 'refit': True})],
    # [FIModelConfig('R2F_lasso', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'lasso', 'max_components_type': 'minnp', 'fraction_chosen': 1.0})],
    # [FIModelConfig('R2F_lasso_bic', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'lasso', 'max_components_type': 'minnp', 'fraction_chosen': 0.9, 'criteria': 'bic', 'refit': False})],
    # [FIModelConfig('R2F_lasso_bic_n2', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'lasso', 'max_components_type': 'minfracnp', 'fraction_chosen': 0.5, 'criteria': 'bic', 'refit': False})],
    # [FIModelConfig('R2F_lasso_bic_refit', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'lasso', 'max_components_type': 'minnp', 'fraction_chosen': 0.9, 'criteria': 'bic', 'refit': True})],
    # [FIModelConfig('R2F_lasso_bic_n2_refit', tree_feature_significance, None, True, model_type='tree', other_params={'type': 'lasso', 'max_components_type': 'minfracnp', 'fraction_chosen': 0.5, 'criteria': 'bic', 'refit': True})],
    # [FIModelConfig('R2F_bic_seq2', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'bic_sequential', 'max_components_type': 'minnp', 'fraction_chosen': 1.0})],
    # [FIModelConfig('R2F_bic_nseq2', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'bic_nonsequential', 'max_components_type': 'minnp', 'fraction_chosen': 1.0})],
    # [FIModelConfig('R2F_adj_bic_nseq2', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'bic_nonsequential', 'max_components_type': 'minnp', 'fraction_chosen': 1.0, 'adjusted_r2': True})],
    # [FIModelConfig('R2F_minnp_1', tree_feature_significance, None, True, model_type='tree',other_params={'max_components_type': 'minnp', 'fraction_chosen': 1.0})],

    # [FIModelConfig('T-Test', lin_reg_t_test, None, True, model_type='linear')],
    # [FIModelConfig('T-Test', lin_reg_marginal_t_test, None, True, model_type='linear')],
    [FIModelConfig('MDI', tree_mdi, None, False, model_type='tree')],
    [FIModelConfig('MDI-oob', tree_mdi_OOB, None, False, model_type='tree')],
    [FIModelConfig('Permutation', perm_importance, None, False, model_type='tree')],
    [FIModelConfig('TreeSHAP', tree_shap_mean, None, False, model_type='tree')],
    # [FIModelConfig('Boruta', boruta_rank, None, False, model_type='rf')],
    # [FIModelConfig('FOCI', foci_rank, None, False, model_type='linear')],  # model_type=None in reality
    # [FIModelConfig('Knockoff', knockpy_swap_integral, None, True, model_type='tree', other_params={'knockoff_fdr':0.05})]
]

# FI_ESTIMATORS = [
#     [FIModelConfig('R2F_minnp_1', tree_feature_significance, None, True, model_type='tree',other_params={'max_components_type': 'minnp', 'fraction_chosen': 1.0})],
#     [FIModelConfig('R2F_bic_seq', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'bic_sequential', 'max_components_type': 'minnp', 'fraction_chosen': 1.0})],
#     [FIModelConfig('R2F_bic_nseq', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'bic_nonsequential', 'max_components_type': 'minnp', 'fraction_chosen': 1.0})],
#     [FIModelConfig('R2F_bic_nseq_bid', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'bic_nonsequential', 'max_components_type': 'minnp', 'fraction_chosen': 1.0, 'direction': 'both'})],
#
#     [FIModelConfig('R2F-_minnp_1', tree_feature_significance, None, True, model_type='tree',other_params={'max_components_type': 'minnp', 'fraction_chosen': 1.0, 'add_linear': False})],
#     [FIModelConfig('R2F-_bic_seq', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'bic_sequential', 'max_components_type': 'minnp', 'fraction_chosen': 1.0, 'add_linear': False})],
#     [FIModelConfig('R2F-_bic_nseq', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'bic_nonsequential', 'max_components_type': 'minnp', 'fraction_chosen': 1.0, 'add_linear': False})],
#     [FIModelConfig('R2F-_bic_nseq_bid', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'bic_nonsequential', 'max_components_type': 'minnp', 'fraction_chosen': 1.0, 'direction': 'both', 'add_linear': False})],
#
#     [FIModelConfig('R2F_adj_minnp_08', tree_feature_significance, None, True, model_type='tree',other_params={'max_components_type': 'minnp', 'fraction_chosen': 0.8, 'adjusted_r2': True})],
#     [FIModelConfig('R2F_adj_bic_nseq', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'bic_nonsequential', 'max_components_type': 'minnp', 'fraction_chosen': 1.0, 'adjusted_r2': True})],
#
#     [FIModelConfig('R2F-_adj_minnp_08', tree_feature_significance, None, True, model_type='tree',other_params={'max_components_type': 'minnp', 'fraction_chosen': 0.8, 'add_linear': False, 'adjusted_r2': True})],
#     [FIModelConfig('R2F-_adj_bic_nseq', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'bic_nonsequential', 'max_components_type': 'minnp', 'fraction_chosen': 1.0, 'add_linear': False, 'adjusted_r2': True})],
#
#     # [FIModelConfig('T-Test', lin_reg_t_test, None, True, model_type='linear')],
#     [FIModelConfig('MDI', tree_mdi, None, False, model_type='tree')],
#     [FIModelConfig('MDI-oob', tree_mdi_OOB, None, False, model_type='tree')],
#     [FIModelConfig('Permutation', perm_importance, None, False, model_type='tree')],
#     [FIModelConfig('TreeSHAP', tree_shap_mean, None, False, model_type='tree')]
# ]

# FI_ESTIMATORS = [
#     [FIModelConfig('R2F_max_norm', tree_feature_significance, None, True, model_type='tree',other_params={'max_components_type': 'max', "normalize": True})],
#     [FIModelConfig('R2F_max', tree_feature_significance, None, True, model_type='tree',other_params={'max_components_type': 'max'})],
#
#     [FIModelConfig('R2F_minnp_1_norm', tree_feature_significance, None, True, model_type='tree',other_params={'max_components_type': 'minnp', 'fraction_chosen': 1.0, "normalize": True})],
#     [FIModelConfig('R2F_minnp_1', tree_feature_significance, None, True, model_type='tree',other_params={'max_components_type': 'minnp', 'fraction_chosen': 1.0})],
#     [FIModelConfig('R2F_minnp_2', tree_feature_significance, None, True, model_type='tree',other_params={'max_components_type': 'minnp', 'fraction_chosen': 0.5})],
#     [FIModelConfig('R2F_minnp_3', tree_feature_significance, None, True, model_type='tree',other_params={'max_components_type': 'minnp', 'fraction_chosen': 0.25})],
#     [FIModelConfig('R2F_minnp_4', tree_feature_significance, None, True, model_type='tree',other_params={'max_components_type': 'minnp', 'fraction_chosen': 0.125})],
#     [FIModelConfig('R2F_minnp_5', tree_feature_significance, None, True, model_type='tree',other_params={'max_components_type': 'minnp', 'fraction_chosen': 1/16})],
#     [FIModelConfig('R2F_minnp_6', tree_feature_significance, None, True, model_type='tree',other_params={'max_components_type': 'minnp', 'fraction_chosen': 1/32})],
#
#     [FIModelConfig('R2F_pca_cv', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'pca_cv', 'max_components_type': 'minnp', 'fraction_chosen': 1.0})],
#
#     [FIModelConfig('R2F_step', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'stepwise', 'max_components_type': 'minnp', 'fraction_chosen': 1.0})],
#
#     [FIModelConfig('R2F_seqstep', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'sequential_stepwise', 'max_components_type': 'minnp', 'fraction_chosen': 1.0})],
#
#     [FIModelConfig('R2F_bic_seq', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'bic_sequential', 'max_components_type': 'minnp', 'fraction_chosen': 1.0})],
#     [FIModelConfig('R2F_bic_nseq', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'bic_nonsequential', 'max_components_type': 'minnp', 'fraction_chosen': 1.0})],
#
#     [FIModelConfig('R2F_pca_var', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'pca_var', 'max_components_type': 'minnp', 'fraction_chosen': 1.0})],
#
#     [FIModelConfig('R2F_ridge', tree_feature_significance, None, True, model_type='tree',other_params={'type': 'ridge', 'max_components_type': "none"})],
#
#     # [FIModelConfig('T-Test', lin_reg_t_test, None, True, model_type='linear')],
#     [FIModelConfig('MDI', tree_mdi, None, False, model_type='tree')],
#     [FIModelConfig('Permutation', perm_importance, None, False, model_type='tree')],
# ]

# FI_ESTIMATORS = [
#     # [FIModelConfig('OptimalTreeSig', optimal_tree_feature_significance, None, True, model_type='tree')],
#     [FIModelConfig('TreeSig', tree_feature_significance, None, True, model_type='tree')],
#     # [FIModelConfig('T-Test', lin_reg_t_test, None, True, model_type='linear')],
#     [FIModelConfig('MDI', tree_mdi, None, False, model_type='tree')],
#     [FIModelConfig('Permutation', perm_importance, None, False, model_type='tree')],
#     [FIModelConfig('TreeSHAP', tree_shap_mean, None, False, model_type='tree')],
#     # [FIModelConfig('Boruta', boruta_rank, None, False, model_type='rf')],
#     # [FIModelConfig('FOCI', foci_rank, None, False, model_type='linear')],  # model_type=None in reality
#     # [FIModelConfig('Knockoff', knockpy_swap_integral, None, True, model_type='tree', other_params={'knockoff_fdr':0.05})]
# ]

# MAX_COMPONENTS=[0.1, 0.2, 0.3, 0.4, 0.5]
# FI_ESTIMATORS = [
#     # [FIModelConfig('OptimalTreeSig', optimal_tree_feature_significance, None, True, model_type='tree', vary_param="max_components", vary_param_val=m) for m in MAX_COMPONENTS],
#     [FIModelConfig('TreeSig', tree_feature_significance, None, True, model_type='tree', vary_param="max_components", vary_param_val=m) for m in MAX_COMPONENTS],
#     # [FIModelConfig('T-Test', lin_reg_t_test, None, True, model_type='linear')],
#     # [FIModelConfig('MDI', tree_mdi, None, False, model_type='tree')],
#     # [FIModelConfig('Permutation', perm_importance, None, False, model_type='tree')],
# ]
