from slurmpy import Slurm

import config
from itertools import product

config_name = 'figs_interactions'
DATASETS_CLASSIFICATION, DATASETS_REGRESSION, \
ESTIMATORS_CLASSIFICATION, ESTIMATORS_REGRESSION = config.get_configs(config_name)
partition = 'high'
# partition = 'low'
s = Slurm("fit", {"partition": partition})


# individual alterations
# DATASETS_CLASSIFICATION = []
# ESTIMATORS_CLASSIFICATION = []


if __name__ == '__main__':
    split_seeds = range(10)
    # est_ds_lst = list(product(DATASETS_CLASSIFICATION, ESTIMATORS_CLASSIFICATION)) \
    #                    + list(product(DATASETS_REGRESSION, ESTIMATORS_REGRESSION))

    est_ds_lst = list(product(DATASETS_REGRESSION, ESTIMATORS_REGRESSION))

    for split_seed in split_seeds:
        for dset, est in est_ds_lst:
            param_str = ''
    #         param_str = 'source ~/chandan/imodels_env/bin/activate; '
            #             param_str += 'module load python'
            param_str += '/accounts/campus/omer_ronen/.conda/envs/imdls_expr/bin/python 01_fit_models.py '
            param_str += f'--dataset "{dset[0]}" '
            param_str += f'--model "{est[0]}" '
            param_str += f'--config {config_name} '
            param_str += f'--split_seed {split_seed} '
            param_str += f'--regression '
    #         param_str += '--ignore_cache'
            s.run(param_str)
            print(param_str)
