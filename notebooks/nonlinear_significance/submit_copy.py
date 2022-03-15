from slurmpy import Slurm

import config
from itertools import product

config_name = 'saps'
DATASETS_CLASSIFICATION, DATASETS_REGRESSION, \
ESTIMATORS_CLASSIFICATION, ESTIMATORS_REGRESSION = config.get_configs(config_name)
partition = 'high'
s = Slurm("fit-imodels-exp", {"partition": partition})

"""
for split_seed in range(0, 6):
    # classification
    for dset in DATASETS_CLASSIFICATION:
        for model in ESTIMATORS_CLASSIFICATION:
            param_str = ''
            param_str += 'python3 01_fit_models.py '
            param_str += f'--dataset "{dset[0]}" '
            param_str += f'--model "{model[0]}" '
            param_str += f'--config {config_name} '
            param_str += f'--split_seed {split_seed} '            
#             param_str += ' '.join(ARGS)
            s.run(param_str)
#             print(param_str)

    # regression
    for dset in DATASETS_REGRESSION:
        for model in ESTIMATORS_REGRESSION:
            param_str = ''
            param_str += 'python3 01_fit_models.py '
            param_str += f'--dataset "{dset[0]}" '
            param_str += f'--model "{model[0]}" '
            param_str += f'--config {config_name} '
            param_str += f'--split_seed {split_seed} '            
#             param_str += ' '.join(ARGS)
            s.run(param_str)
#             print(param_str)
"""


for split_seed in range(1, 6):
    for dset, est in list(product(DATASETS_CLASSIFICATION, ESTIMATORS_CLASSIFICATION)) \
                   + list(product(DATASETS_REGRESSION, ESTIMATORS_REGRESSION)):
        param_str = ''
#         param_str = 'source ~/chandan/imodels_env/bin/activate; '
        #             param_str += 'module load python'
        param_str += 'python3 01_fit_models.py '
        param_str += f'--dataset "{dset[0]}" '
        param_str += f'--model "{est[0]}" '
        param_str += f'--config {config_name} '
        param_str += f'--split_seed {split_seed} '            
        s.run(param_str)
        print(param_str)