from slurmpy import Slurm

import config
from itertools import product

config_name = 'figs_ensembles'
DATASETS_CLASSIFICATION, DATASETS_REGRESSION, \
ESTIMATORS_CLASSIFICATION, ESTIMATORS_REGRESSION = config.get_configs(config_name)
split_seeds = range(1)
partition = 'high'
# partition = 'low'
s = Slurm("fit", {"partition": partition})

for split_seed in split_seeds:
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
        param_str += '--ignore_cache'
        s.run(param_str)
        print(param_str)