import argparse
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
        # print('No files found at ', path)
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

    if 'df_rules' in results_sorted[0]:
        rule_df = pd.concat([r['df_rules'] for r in results_sorted])
        output_dict['df_rules'] = rule_df

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
        fname = oj(path, seed_path, 'results_aggregated.pkl')
        if os.path.exists(fname):  # check that this seed has actually saved
            results_seed = pkl.load(open(fname, 'rb'))
            for k in results_seed.keys():
                if 'df' not in k:
                    results_overall[k] = results_seed[k]
                else:
                    if k in results_overall:
                        # print(results_seed[k]['split_seed'])
                        results_overall[k] = pd.concat((results_overall[k], results_seed[k])).reset_index(drop=True)
                    else:
                        results_overall[k] = results_seed[k]

    # keys to aggregate over

    if 'df' in results_overall:
        ks = []
        df = results_overall['df']
        mets = results_overall['metrics']
        for k in df.keys():
            skip = False
            for met in mets:
                if k.startswith(met):
                    skip = True
            if not skip:
                ks.append(k)
        ks.remove('split_seed')

        # mean
        grouped = df.fillna(-1).groupby(ks)
        dm = grouped.mean().drop(columns='split_seed')
        ds = grouped.sem().drop(columns='split_seed')
        results_overall['df_mean'] = dm.join(ds, lsuffix='_mean', rsuffix='_std').reset_index()

    pkl.dump(results_overall, open(oj(path, 'results_aggregated.pkl'), 'wb'))
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str,
                        default=oj(os.path.dirname(os.path.realpath(__file__)), 'results'))
    args = parser.parse_args()
    results_root = args.results_path

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
        if os.path.isdir(result_path):
            successful += aggregate_over_seeds(result_path)
            total += 1
    print('successfully processed', successful, '/', total, 'averaged seeds')
