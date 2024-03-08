from sklearn.ensemble import RandomForestRegressor
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods_local import *
# N_ESTIMATORS=[50, 100, 500, 1000]
ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='tree',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33})],
    # [ModelConfig('RF', RandomForestRegressor, model_type='tree', vary_param="n_estimators", vary_param_val=m,
    #              other_params={'min_samples_leaf': 5, 'max_features': 0.33}) for m in N_ESTIMATORS]
]

FI_ESTIMATORS = [
    [FIModelConfig('MDI_local_all_stumps_evaluate', MDI_local_all_stumps_evaluate, ascending = False, splitting_strategy = "train-test", model_type='tree')],
    # [FIModelConfig('MDI_sub_stumps', MDI_local_sub_stumps, ascending = False, model_type='tree')],
    [FIModelConfig('MDI_local_all_stumps_evaluate_without_raw', MDI_local_all_stumps_evaluate, ascending = False, splitting_strategy = "train-test", model_type='tree', other_params={"include_raw": False})],
    # [FIModelConfig('MDI_sub_stumps_without_raw', MDI_local_sub_stumps, ascending = False, model_type='tree', other_params={"include_raw": False})],
    # [FIModelConfig('LFI_sum_absolute', LFI_sum_absolute, model_type='tree', splitting_strategy = "train-test")],
    [FIModelConfig('LFI_absolute_sum_evaluate', LFI_absolute_sum_evaluate, model_type='tree', splitting_strategy = "train-test")],
    # [FIModelConfig('LFI_sum_absolute_sub_stumps', LFI_sum_absolute_sub_stumps, model_type='tree')],
    # [FIModelConfig('LFI_absolute_sum_sub_stumps', LFI_absolute_sum_sub_stumps, model_type='tree')],
    # [FIModelConfig('LFI_sum_absolute_without_raw', LFI_sum_absolute, model_type='tree', splitting_strategy = "train-test", other_params={"include_raw": False})],
    [FIModelConfig('LFI_absolute_sum_evaluate_without_raw', LFI_absolute_sum_evaluate, model_type='tree', splitting_strategy = "train-test", other_params={"include_raw": False})],
    # [FIModelConfig('LFI_sum_absolute_sub_stumps_without_raw', LFI_sum_absolute_sub_stumps, model_type='tree', other_params={"include_raw": False})],
    # [FIModelConfig('LFI_absolute_sum_sub_stumps_without_raw', LFI_absolute_sum_sub_stumps, model_type='tree', other_params={"include_raw": False})],
    [FIModelConfig('TreeSHAP', tree_shap_local, model_type='tree', splitting_strategy = "train-test")],
    [FIModelConfig('LIME', lime_local, model_type='tree', splitting_strategy = "train-test")],
]

# [FIModelConfig('Permutation', permutation_local, model_type='tree')],