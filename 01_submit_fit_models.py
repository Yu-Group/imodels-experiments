from slurmpy import Slurm

import config

EXP_TYPE = 'saps'
MODEL = 'C45'
ARGS = ['']

DATASETS_CLASSIFICATION, DATASETS_REGRESSION, \
ESTIMATORS_CLASSIFICATION, ESTIMATORS_REGRESSION = config.get_configs(EXP_TYPE)
DATASETS_ALL = DATASETS_CLASSIFICATION + DATASETS_REGRESSION
ESTIMATORS_ALL = [*ESTIMATORS_CLASSIFICATION, *ESTIMATORS_REGRESSION]

partition = 'high'
s = Slurm("compare_models", {"partition": partition})

# for dset in DATASETS_ALL:
#     for est in ESTIMATORS_ALL:
#         param_str = 'source ~/chandan/imodels_env/bin/activate; '
#         param_str += 'python3 ~/chandan/rules-experiments/01_run_comparisons.py '
#         param_str += f'--dataset "{dset[0]}" '
#         param_str += f'--model "{est[0].name}" '
#         s.run(param_str)


for dset in DATASETS_ALL:
    # param_str = 'source ~/chandan/imodels_env/bin/activate; '
    
    param_str = 'python3 01_run_comparisons.py '
    param_str += f'--dataset "{dset[0]}" '
    param_str += f'--model "{MODEL}" '
    param_str += f'--config {EXP_TYPE} '
    param_str += ' '.join(ARGS)
    s.run(param_str)
