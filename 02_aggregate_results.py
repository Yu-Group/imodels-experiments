import glob
import os.path
import pickle as pkl
import warnings
from os.path import join as oj

import numpy as np
import pandas as pd
from tqdm import tqdm

from validate import compute_meta_auc


def aggregate_single_seed(path: str):
    """Combines comparisons output after running
    Parameters
    ----------
    path: str
        path to directory containing pkl files to combine
    """
    # print('path', '/'.join(path.split('/')[-4:]))
    all_files = glob.glob(oj(path, '*'))
    model_files = sorted([f for f in all_files if '_comparisons' in f])

    if len(model_files) == 0:
        print('No files found at ', path)
        return 0

    # print('\tprocessing path', '/'.join(path.split('/')[-4:]))
    results_sorted = [pkl.load(open(f, 'rb')) for f in model_files]

    df = pd.concat([r['df'] for r in results_sorted])
    estimators = []
    for r in results_sorted:
        estimators += np.unique(r['estimators']).tolist()

    output_dict = {
        'estimators': estimators,
        'comparison_datasets': results_sorted[0]['comparison_datasets'],
        'metrics': results_sorted[0]['metrics'],
        'df': df,
    }

    if 'rule_df' in results_sorted[0]:
        rule_df = pd.concat([r['rule_df'] for r in results_sorted])
        output_dict['rule_df'] = rule_df

    # for curr_df, prefix in level_dfs:
    try:
        meta_auc_df = compute_meta_auc(df)
    except Exception as e:
        warnings.warn(f'bad complexity range')
        # warnings.warn(e)
        meta_auc_df = None

    output_dict['meta_auc_df'] = meta_auc_df
    # combined_filename = '.'.join(model_files_sorted[0].split('_0.'))
    # pkl.dump(output_dict, open(combined_filename, 'wb'))

    combined_filename = oj(path, 'results_aggregated.pkl')
    pkl.dump(output_dict, open(combined_filename, 'wb'))
    return 1


def aggregate_over_seeds(path: str):
    """Combine pkl files across many seeds
    """
    results_overall = {}
    for seed_path in os.listdir(path):
        try:
            results_seed = pkl.load(open(oj(path, seed_path, 'results_aggregated.pkl'), 'rb'))
            for k in results_seed.keys():
                if 'df' not in k:
                    results_overall[k] = results_seed[k]
                else:
                    if k in results_overall:
                        # print(results_seed[k]['split_seed'])
                        results_overall[k] = pd.concat((results_overall[k], results_seed[k])).reset_index()
                    else:
                        results_overall[k] = results_seed[k]
        except:
            pass
    pkl.dump(results_overall, open(oj(path, 'results_aggregated.pkl'), 'wb'))
    return 1


if __name__ == "__main__":
    results_root = oj(os.path.dirname(os.path.realpath(__file__)), 'results')

    # results/config_name/dataset/splitting strategy/seednum/*.pkl
    successful = 0
    total = 0
    for result_path in tqdm(glob.glob(f'{results_root}/*/*/*/*')):
        successful += aggregate_single_seed(result_path)
        total += 1
    print('successfully processed', successful, '/', total, 'individual seeds')

    successful = 0
    total = 0
    for result_path in tqdm(glob.glob(f'{results_root}/*/*/*')):
        successful += aggregate_over_seeds(result_path)
        total += 1
    print('successfully processed', successful, '/', total, 'averaged seeds')
