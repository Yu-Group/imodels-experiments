import importlib

def get_fi_configs(config_name):
    ests = importlib.import_module(f'fi_config.{config_name}.models')
    dgp = importlib.import_module(f'fi_config.{config_name}.dgp')
    return ests.ESTIMATORS, ests.FI_ESTIMATORS, \
           dgp.X_DGP, dgp.X_PARAMS_DICT, dgp.Y_DGP, dgp.Y_PARAMS_DICT, \
           dgp.VARY_PARAM_NAME, dgp.VARY_PARAM_VALS