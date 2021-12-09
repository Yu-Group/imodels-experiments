from slurmpy import Slurm

import config

config_name = 'saps'
# MODEL = 'C45'
# ARGS = ['']

DATASETS_CLASSIFICATION, DATASETS_REGRESSION, \
ESTIMATORS_CLASSIFICATION, ESTIMATORS_REGRESSION = config.get_configs(config_name)
# DATASETS_ALL = DATASETS_CLASSIFICATION + DATASETS_REGRESSION
# ESTIMATORS_ALL = ESTIMATORS_CLASSIFICATION + ESTIMATORS_REGRESSION

partition = 'high'
s = Slurm("compare_models", {"partition": partition})


for split_seed in range(3, 6):
    
    # classification
    for dset in DATASETS_CLASSIFICATION:
        for model in ESTIMATORS_CLASSIFICATION:
        # param_str = 'source ~/chandan/imodels_env/bin/activate; '
            param_str = ''
#             param_str += 'module load python'
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
        # param_str = 'source ~/chandan/imodels_env/bin/activate; '
            param_str = ''
#             param_str += 'module load python'
            param_str += 'python3 01_fit_models.py '
            param_str += f'--dataset "{dset[0]}" '
            param_str += f'--model "{model[0]}" '
            param_str += f'--config {config_name} '
            param_str += f'--split_seed {split_seed} '            
#             param_str += ' '.join(ARGS)
            s.run(param_str)
#             print(param_str)



# for dset in DATASETS_ALL:
#     for est in ESTIMATORS_ALL:
#         param_str = 'source ~/chandan/imodels_env/bin/activate; '
#         param_str += 'python3 ~/chandan/rules-experiments/01_run_comparisons.py '
#         param_str += f'--dataset "{dset[0]}" '
#         param_str += f'--model "{est[0].name}" '
#         s.run(param_str)