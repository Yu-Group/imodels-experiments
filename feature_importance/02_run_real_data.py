# Example usage: run in command line
# cd feature_importance/
# python 01_run_simulations.py --nreps 2 --config test --split_seed 12345 --ignore_cache
# python 01_run_simulations.py --nreps 2 --config test --split_seed 12345 --ignore_cache --create_rmd

import copy
import os
from os.path import join as oj
import glob
import argparse
import pickle as pkl
import time
import warnings
from scipy import stats
import dask
from dask.distributed import Client
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from collections import defaultdict
from typing import Callable, List, Tuple
import itertools

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
import fi_config
from util import ModelConfig, FIModelConfig, apply_splitting_strategy

warnings.filterwarnings("ignore", message="Bins whose width")


def compare_estimators(estimators: List[ModelConfig],
                       fi_estimators: List[FIModelConfig],
                       X, y, args, ) -> Tuple[dict, dict]:
    """Calculates results given estimators, feature importance estimators, and datasets.
    Called in run_comparison
    """
    if type(estimators) != list:
        raise Exception("First argument needs to be a list of Models")

    # initialize results
    results = defaultdict(lambda: [])

    # loop over model estimators
    for model in tqdm(estimators, leave=False):
        est = model.cls(**model.kwargs)

        # get kwargs for all fi_ests
        fi_kwargs = {}
        for fi_est in fi_estimators:
            fi_kwargs.update(fi_est.kwargs)

        # get groups of estimators for each splitting strategy
        fi_ests_dict = defaultdict(list)
        for fi_est in fi_estimators:
            fi_ests_dict[fi_est.splitting_strategy].append(fi_est)

        # loop over splitting strategies
        for splitting_strategy, fi_ests in fi_ests_dict.items():
            # implement provided splitting strategy
            if splitting_strategy is not None:
                X_train, X_tune, X_test, y_train, y_tune, y_test = apply_splitting_strategy(X, y, splitting_strategy, args.split_seed)
            else:
                X_train = X
                X_tune = X
                X_test = X
                y_train = y
                y_tune = y
                y_test = y

            # fit model
            est.fit(X_train, y_train)

            # loop over fi estimators
            for fi_est in fi_ests:
                metric_results = {
                    'model': model.name,
                    'fi': fi_est.name,
                    'splitting_strategy': splitting_strategy
                }
                start = time.time()
                fi_score = fi_est.cls(X_test, y_test, copy.deepcopy(est), **fi_est.kwargs)
                end = time.time()
                metric_results['fi_scores'] = copy.deepcopy(fi_score)
                metric_results['time'] = end - start

                # initialize results with metadata and results
                kwargs: dict = model.kwargs  # dict
                for k in kwargs:
                    results[k].append(kwargs[k])
                for k in fi_kwargs:
                    if k in fi_est.kwargs:
                        results[k].append(fi_est.kwargs[k])
                    else:
                        results[k].append(None)
                for met_name, met_val in metric_results.items():
                    results[met_name].append(met_val)
    return results


def run_comparison(path: str,
                   X, y,
                   estimators: List[ModelConfig],
                   fi_estimators: List[FIModelConfig],
                   args):
    estimator_name = estimators[0].name.split(' - ')[0]
    fi_estimators_all = [fi_estimator for fi_estimator in itertools.chain(*fi_estimators) \
                         if fi_estimator.model_type in estimators[0].model_type]
    model_comparison_files_all = [oj(path, f'{estimator_name}_{fi_estimator.name}_comparisons.pkl') \
                                  for fi_estimator in fi_estimators_all]
    if args.parallel_id is not None:
        model_comparison_files_all = [f'_{args.parallel_id[0]}.'.join(model_comparison_file.split('.')) \
                                      for model_comparison_file in model_comparison_files_all]

    fi_estimators = []
    model_comparison_files = []
    for model_comparison_file, fi_estimator in zip(model_comparison_files_all, fi_estimators_all):
        if os.path.isfile(model_comparison_file) and not args.ignore_cache:
            print(
                f'{estimator_name} with {fi_estimator.name} results already computed and cached. use --ignore_cache to recompute')
        else:
            fi_estimators.append(fi_estimator)
            model_comparison_files.append(model_comparison_file)

    if len(fi_estimators) == 0:
        return

    results = compare_estimators(estimators=estimators,
                                 fi_estimators=fi_estimators,
                                 X=X, y=y,
                                 args=args)

    estimators_list = [e.name for e in estimators]

    df = pd.DataFrame.from_dict(results)
    df['split_seed'] = args.split_seed

    for model_comparison_file, fi_estimator in zip(model_comparison_files, fi_estimators):
        output_dict = {
            # metadata
            'sim_name': args.config,
            'estimators': estimators_list,
            'fi_estimators': fi_estimator.name,

            # actual values
            'df': df.loc[df.fi == fi_estimator.name],
        }
        pkl.dump(output_dict, open(model_comparison_file, 'wb'))
    return df


def reformat_results(results):
    results = results.reset_index().drop(columns=['index'])
    fi_scores = pd.concat(results.pop('fi_scores').to_dict()). \
        reset_index(level=0).rename(columns={'level_0': 'index'})
    results_df = pd.merge(results, fi_scores, left_index=True, right_on="index")
    return results_df


def run_simulation(i, path, Xpath, ypath, ests, fi_ests, args):
    X_df = pd.read_csv(Xpath)
    y_df = pd.read_csv(ypath)
    for col in y_df.columns:
        y = y_df[col].to_numpy().ravel()
        keep_idx = ~pd.isnull(y)
        X = X_df[keep_idx].to_numpy()
        y = y[keep_idx]
        if y_df.shape[1] > 1:
            output_path = oj(path, col)
        else:
            output_path = path
        os.makedirs(oj(output_path, "rep" + str(i)), exist_ok=True)
        for est in ests:
            for idx in range(len(est)):
                if "random_state" in est[idx].kwargs.keys():
                    est[idx].kwargs["random_state"] = i
            results = run_comparison(
                path=oj(output_path, "rep" + str(i)),
                X=X, y=y,
                estimators=est,
                fi_estimators=fi_ests,
                args=args
            )

    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    default_dir = os.getenv("SCRATCH")
    if default_dir is not None:
        default_dir = oj(default_dir, "feature_importance", "results")
    else:
        default_dir = oj(os.path.dirname(os.path.realpath(__file__)), 'results')

    parser.add_argument('--nreps', type=int, default=1)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--fi_model', type=str, default=None)
    parser.add_argument('--config', type=str, default='test_real_data')

    # for multiple reruns, should support varying split_seed
    parser.add_argument('--ignore_cache', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--parallel_id', nargs='+', type=int, default=None)
    parser.add_argument('--n_cores', type=int, default=None)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--results_path', type=str, default=default_dir)

    args = parser.parse_args()

    if args.parallel:
        if args.n_cores is None:
            print(os.getenv("SLURM_CPUS_ON_NODE"))
            n_cores = int(os.getenv("SLURM_CPUS_ON_NODE"))
        else:
            n_cores = args.n_cores
        client = Client(n_workers=n_cores)

    ests, fi_ests, Xpath, ypath = fi_config.get_fi_configs(args.config, real_data=True)

    if args.model:
        ests = list(filter(lambda x: args.model.lower() == x[0].name.lower(), ests))
    if args.fi_model:
        fi_ests = list(filter(lambda x: args.fi_model.lower() == x[0].name.lower(), fi_ests))

    if len(ests) == 0:
        raise ValueError('No valid estimators', 'sim', args.config, 'models', args.model, 'fi', args.fi_model)
    if len(fi_ests) == 0:
        raise ValueError('No valid FI estimators', 'sim', args.config, 'models', args.model, 'fi', args.fi_model)
    if args.verbose:
        print('running', args.config,
              'ests', ests,
              'fi_ests', fi_ests)
        print('\tsaving to', args.results_path)

    results_dir = oj(args.results_path, args.config)
    path = oj(results_dir, "seed" + str(args.split_seed))
    os.makedirs(path, exist_ok=True)

    eval_out = defaultdict(list)

    if args.parallel:
        futures = [dask.delayed(run_simulation)(i, path, Xpath, ypath, ests, fi_ests, args) for i in range(args.nreps)]
        results = dask.compute(*futures)
    else:
        results = [run_simulation(i, path, Xpath, ypath, ests, fi_ests, args) for i in range(args.nreps)]
    assert all(results)

    print('completed all experiments successfully!')

    # get model file names
    model_comparison_files_all = []
    for est in ests:
        estimator_name = est[0].name.split(' - ')[0]
        fi_estimators_all = [fi_estimator for fi_estimator in itertools.chain(*fi_ests) \
                             if fi_estimator.model_type in est[0].model_type]
        model_comparison_files = [f'{estimator_name}_{fi_estimator.name}_comparisons.pkl' for fi_estimator in
                                  fi_estimators_all]
        model_comparison_files_all += model_comparison_files

    # aggregate results
    y_df = pd.read_csv(ypath)
    results_list = []
    for col in y_df.columns:
        if y_df.shape[1] > 1:
            output_path = oj(path, col)
        else:
            output_path = path
        for i in range(args.nreps):
            all_files = glob.glob(oj(output_path, 'rep' + str(i), '*'))
            model_files = sorted([f for f in all_files if os.path.basename(f) in model_comparison_files_all])

            if len(model_files) == 0:
                print('No files found at ', oj(output_path, 'rep' + str(i)))
                continue

            results = pd.concat(
                [pkl.load(open(f, 'rb'))['df'] for f in model_files],
                axis=0
            )
            results.insert(0, 'rep', i)
            if y_df.shape[1] > 1:
                results.insert(1, 'y_task', col)
            results_list.append(results)

    results_merged = pd.concat(results_list, axis=0)
    pkl.dump(results_merged, open(oj(path, 'results.pkl'), 'wb'))
    results_df = reformat_results(results_merged)
    results_df.to_csv(oj(path, 'results.csv'), index=False)

    print('merged and saved all experiment results successfully!')

# %%
