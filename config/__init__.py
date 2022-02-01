import importlib
X = 'val'

def get_configs(config_name):
    dsets = importlib.import_module(f'config.{config_name}.datasets')
    ests = importlib.import_module(f'config.{config_name}.models')
    return dsets.DATASETS_CLASSIFICATION, dsets.DATASETS_REGRESSION, ests.ESTIMATORS_CLASSIFICATION, ests.ESTIMATORS_REGRESSION
