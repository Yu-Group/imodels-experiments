# Example usage: run in command line
# cd feature_importance/
# python 03_run_real_data_prediction.py --nreps 2 --config test --split_seed 12345 --ignore_cache
# python 03_run_real_data_prediction.py --nreps 2 --config test --split_seed 12345 --ignore_cache --create_rmd

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
from functools import partial

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
import fi_config
from util import ModelConfig, apply_splitting_strategy, auroc_score, auprc_score

from sklearn.metrics import accuracy_score, f1_score, recall_score, \
    precision_score, average_precision_score, r2_score, explained_variance_score, \
    mean_squared_error, mean_absolute_error, log_loss

warnings.filterwarnings("ignore", message="Bins whose width")


def compare_estimators(estimators: List[ModelConfig],
                       X, y,
                       metrics: List[Tuple[str, Callable]],
                       args, rep) -> Tuple[dict, dict]:
    """Calculates results given estimators, feature importance estimators, and datasets.
    Called in run_comparison
    """
    if type(estimators) != list:
        raise Exception("First argument needs to be a list of Models")
    if type(metrics) != list:
        raise Exception("Argument metrics needs to be a list containing ('name', callable) pairs")

    # initialize results
    results = defaultdict(lambda: [])

    if args.splitting_strategy is not None:
        X_train, X_tune, X_test, y_train, y_tune, y_test = apply_splitting_strategy(
            X, y, args.splitting_strategy, args.split_seed + rep)
    else:
        X_train = X
        X_tune = X
        X_test = X
        y_train = y
        y_tune = y
        y_test = y

    # loop over model estimators
    for model in tqdm(estimators, leave=False):
        est = model.cls(**model.kwargs)

        start = time.time()
        est.fit(X_train, y_train)
        end = time.time()

        metric_results = {'model': model.name}
        y_pred = est.predict(X_test)
        if args.mode != 'regression':
            y_pred_proba = est.predict_proba(X_test)
            if args.mode == 'binary_classification':
                y_pred_proba = y_pred_proba[:, 1]
        else:
            y_pred_proba = y_pred
        for met_name, met in metrics:
            if met is not None:
                if args.mode == 'regression' \
                        or met_name in ['accuracy', 'f1', 'precision', 'recall']:
                    metric_results[met_name] = met(y_test, y_pred)
                else:
                    metric_results[met_name] = met(y_test, y_pred_proba)
        metric_results['predictions'] = copy.deepcopy(pd.DataFrame(y_pred_proba))
        metric_results['time'] = end - start

        # initialize results with metadata and metric results
        kwargs: dict = model.kwargs  # dict
        for k in kwargs:
            results[k].append(kwargs[k])
        for met_name, met_val in metric_results.items():
            results[met_name].append(met_val)

    return results


def run_comparison(rep: int,
                   path: str,
                   X, y,
                   metrics: List[Tuple[str, Callable]],
                   estimators: List[ModelConfig],
                   args):
    estimator_name = estimators[0].name.split(' - ')[0]
    model_comparison_file = oj(path, f'{estimator_name}_comparisons.pkl')
    if args.parallel_id is not None:
        model_comparison_file = f'_{args.parallel_id[0]}.'.join(model_comparison_file.split('.'))

    if os.path.isfile(model_comparison_file) and not args.ignore_cache:
        print(f'{estimator_name} results already computed and cached. use --ignore_cache to recompute')
        return

    results = compare_estimators(estimators=estimators,
                                 X=X, y=y,
                                 metrics=metrics,
                                 args=args,
                                 rep=rep)

    estimators_list = [e.name for e in estimators]

    df = pd.DataFrame.from_dict(results)
    df['split_seed'] = args.split_seed + rep
    if args.nosave_cols is not None:
        nosave_cols = np.unique([x.strip() for x in args.nosave_cols.split(",")])
    else:
        nosave_cols = []
    for col in nosave_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    output_dict = {
        # metadata
        'sim_name': args.config,
        'estimators': estimators_list,

        # actual values
        'df': df,
    }
    pkl.dump(output_dict, open(model_comparison_file, 'wb'))

    return df


def reformat_results(results):
    results = results.reset_index().drop(columns=['index'])
    predictions = pd.concat(results.pop('predictions').to_dict()). \
        reset_index(level=[0, 1]).rename(columns={'level_0': 'index', 'level_1': 'sample_id'})
    results_df = pd.merge(results, predictions, left_index=True, right_on="index")
    return results_df


def get_metrics(mode: str = 'regression'):
    if mode == 'binary_classification':
        return [
            ('rocauc', auroc_score),
            ('prauc', auprc_score),
            ('logloss', log_loss),
            ('accuracy', accuracy_score),
            ('f1', f1_score),
            ('recall', recall_score),
            ('precision', precision_score),
            ('avg_precision', average_precision_score)
        ]
    elif mode == 'multiclass_classification':
        return [
            ('rocauc', partial(auroc_score, multi_class="ovr")),
            ('prauc', partial(auprc_score, multi_class="ovr")),
            ('logloss', log_loss),
            ('accuracy', accuracy_score),
            ('f1', partial(f1_score, average='micro')),
            ('recall', partial(recall_score, average='micro')),
            ('precision', partial(precision_score, average='micro'))
        ]
    elif mode == 'regression':
        return [
            ('r2', r2_score),
            ('explained_variance', explained_variance_score),
            ('mean_squared_error', mean_squared_error),
            ('mean_absolute_error', mean_absolute_error),
        ]


def run_simulation(i, path, Xpath, ypath, ests, metrics, args):
    X_df = pd.read_csv(Xpath)
    y_df = pd.read_csv(ypath)
    if args.subsample_n is not None:
        if args.subsample_n < X_df.shape[0]:
            keep_rows = np.random.choice(X_df.shape[0], args.subsample_n, replace=False)
            X_df = X_df.iloc[keep_rows]
            y_df = y_df.iloc[keep_rows]
    if args.response_idx is None:
        keep_cols = y_df.columns
    else:
        keep_cols = [args.response_idx]
    for col in keep_cols:
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
                rep=i,
                path=oj(output_path, "rep" + str(i)),
                X=X, y=y,
                metrics=metrics,
                estimators=est,
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

    parser.add_argument('--nreps', type=int, default=2)
    parser.add_argument('--mode', type=str, default='regression')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--config', type=str, default='gmdi.prediction_sims.ccle_rnaseq_regression-')
    parser.add_argument('--response_idx', type=str, default=None)
    parser.add_argument('--subsample_n', type=int, default=None)
    parser.add_argument('--nosave_cols', type=str, default=None)

    # for multiple reruns, should support varying split_seed
    parser.add_argument('--ignore_cache', action='store_true', default=False)
    parser.add_argument('--splitting_strategy', type=str, default="train-test")
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--parallel_id', nargs='+', type=int, default=None)
    parser.add_argument('--n_cores', type=int, default=None)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--results_path', type=str, default=default_dir)

    args = parser.parse_args()

    assert args.splitting_strategy in {
        'train-test', 'train-tune-test', 'train-test-lowdata', 'train-tune-test-lowdata'}
    assert args.mode in {'regression', 'binary_classification', 'multiclass_classification'}

    if args.parallel:
        if args.n_cores is None:
            print(os.getenv("SLURM_CPUS_ON_NODE"))
            n_cores = int(os.getenv("SLURM_CPUS_ON_NODE"))
        else:
            n_cores = args.n_cores
        client = Client(n_workers=n_cores)

    ests, _, Xpath, ypath = fi_config.get_fi_configs(args.config, real_data=True)
    metrics = get_metrics(args.mode)

    if args.model:
        ests = list(filter(lambda x: args.model.lower() == x[0].name.lower(), ests))

    if len(ests) == 0:
        raise ValueError('No valid estimators', 'sim', args.config, 'models', args.model)
    if args.verbose:
        print('running', args.config,
              'ests', ests)
        print('\tsaving to', args.results_path)

    results_dir = oj(args.results_path, args.config)
    path = oj(results_dir, "seed" + str(args.split_seed))
    os.makedirs(path, exist_ok=True)

    eval_out = defaultdict(list)

    if args.parallel:
        futures = [dask.delayed(run_simulation)(i, path, Xpath, ypath, ests, metrics, args) for i in range(args.nreps)]
        results = dask.compute(*futures)
    else:
        results = [run_simulation(i, path, Xpath, ypath, ests, metrics, args) for i in range(args.nreps)]
    assert all(results)

    print('completed all experiments successfully!')

    # get model file names
    model_comparison_files_all = []
    for est in ests:
        estimator_name = est[0].name.split(' - ')[0]
        model_comparison_file = f'{estimator_name}_comparisons.pkl'
        model_comparison_files_all.append(model_comparison_file)

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
