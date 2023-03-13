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


def run_simulation(i, path, val_name, X_params_dict, X_dgp, y_params_dict, y_dgp, ests, metrics, args):
    os.makedirs(oj(path, val_name, "rep" + str(i)), exist_ok=True)
    np.random.seed(i)
    max_iter = 100
    iter = 0
    while iter <= max_iter:  # regenerate data if y is constant
        X = X_dgp(**X_params_dict)
        y, support, beta = y_dgp(X, **y_params_dict, return_support=True)
        if not all(y == y[0]):
            break
        iter += 1
    if iter > max_iter:
        raise ValueError("Response y is constant.")
    if args.omit_vars is not None:
        omit_vars = np.unique([int(x.strip()) for x in args.omit_vars.split(",")])
        support = np.delete(support, omit_vars)
        X = np.delete(X, omit_vars, axis=1)
        del beta  # note: beta is not currently supported when using omit_vars

    for est in ests:
        results = run_comparison(rep=i,
                                 path=oj(path, val_name, "rep" + str(i)),
                                 X=X, y=y,
                                 metrics=metrics,
                                 estimators=est,
                                 args=args)
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
    parser.add_argument('--omit_vars', type=str, default=None)  # comma-separated string of variables to omit
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

    ests, fi_ests, \
    X_dgp, X_params_dict, y_dgp, y_params_dict, \
    vary_param_name, vary_param_vals = fi_config.get_fi_configs(args.config)
    metrics = get_metrics(args.mode)

    if args.model:
        ests = list(filter(lambda x: args.model.lower() == x[0].name.lower(), ests))

    if len(ests) == 0:
        raise ValueError('No valid estimators', 'sim', args.config, 'models', args.model)
    if args.verbose:
        print('running', args.config,
              'ests', ests)
        print('\tsaving to', args.results_path)

    if args.omit_vars is not None:
        results_dir = oj(args.results_path, args.config + "_omitted_vars")
    else:
        results_dir = oj(args.results_path, args.config)

    if isinstance(vary_param_name, list):
        path = oj(results_dir, "varying_" + "_".join(vary_param_name), "seed" + str(args.split_seed))
    else:
        path = oj(results_dir, "varying_" + vary_param_name, "seed" + str(args.split_seed))
    os.makedirs(path, exist_ok=True)

    eval_out = defaultdict(list)

    vary_type = None
    if isinstance(vary_param_name, list):  # multiple parameters are being varied
        # get parameters that are being varied over and identify whether it's a DGP/method/fi_method argument
        keys, values = zip(*vary_param_vals.items())
        vary_param_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        vary_type = {}
        for vary_param_dict in vary_param_dicts:
            for param_name, param_val in vary_param_dict.items():
                if param_name in X_params_dict.keys() and param_name in y_params_dict.keys():
                    raise ValueError('Cannot vary over parameter in both X and y DGPs.')
                elif param_name in X_params_dict.keys():
                    vary_type[param_name] = "dgp"
                    X_params_dict[param_name] = vary_param_vals[param_name][param_val]
                elif param_name in y_params_dict.keys():
                    vary_type[param_name] = "dgp"
                    y_params_dict[param_name] = vary_param_vals[param_name][param_val]
                else:
                    est_kwargs = list(
                        itertools.chain(*[list(est.kwargs.keys()) for est in list(itertools.chain(*ests))]))
                    if param_name in est_kwargs:
                        vary_type[param_name] = "est"
                    else:
                        raise ValueError('Invalid vary_param_name.')

            if args.parallel:
                futures = [
                    dask.delayed(run_simulation)(i, path, "_".join(vary_param_dict.values()), X_params_dict, X_dgp,
                                                 y_params_dict, y_dgp, ests, metrics, args) for i in
                    range(args.nreps)]
                results = dask.compute(*futures)
            else:
                results = [
                    run_simulation(i, path, "_".join(vary_param_dict.values()), X_params_dict, X_dgp, y_params_dict,
                                   y_dgp, ests, metrics, args) for i in range(args.nreps)]
            assert all(results)

    else:  # only on parameter is being varied over
        # get parameter that is being varied over and identify whether it's a DGP/method/fi_method argument
        for val_name, val in vary_param_vals.items():
            if vary_param_name in X_params_dict.keys() and vary_param_name in y_params_dict.keys():
                raise ValueError('Cannot vary over parameter in both X and y DGPs.')
            elif vary_param_name in X_params_dict.keys():
                vary_type = "dgp"
                X_params_dict[vary_param_name] = val
            elif vary_param_name in y_params_dict.keys():
                vary_type = "dgp"
                y_params_dict[vary_param_name] = val
            else:
                est_kwargs = list(itertools.chain(*[list(est.kwargs.keys()) for est in list(itertools.chain(*ests))]))
                if vary_param_name in est_kwargs:
                    vary_type = "est"
                else:
                    raise ValueError('Invalid vary_param_name.')

            if args.parallel:
                futures = [
                    dask.delayed(run_simulation)(i, path, val_name, X_params_dict, X_dgp, y_params_dict, y_dgp, ests,
                                                 metrics, args) for i in range(args.nreps)]
                results = dask.compute(*futures)
            else:
                results = [run_simulation(i, path, val_name, X_params_dict, X_dgp, y_params_dict, y_dgp, ests, 
                                          metrics, args) for i in range(args.nreps)]
            assert all(results)

    print('completed all experiments successfully!')

    # get model file names
    model_comparison_files_all = []
    for est in ests:
        estimator_name = est[0].name.split(' - ')[0]
        model_comparison_file = f'{estimator_name}_comparisons.pkl'
        model_comparison_files_all.append(model_comparison_file)

    # aggregate results
    # aggregate results
    results_list = []
    if isinstance(vary_param_name, list):
        for vary_param_dict in vary_param_dicts:
            val_name = "_".join(vary_param_dict.values())

            for i in range(args.nreps):
                all_files = glob.glob(oj(path, val_name, 'rep' + str(i), '*'))
                model_files = sorted([f for f in all_files if os.path.basename(f) in model_comparison_files_all])

                if len(model_files) == 0:
                    print('No files found at ', oj(path, val_name, 'rep' + str(i)))
                    continue

                results = pd.concat(
                    [pkl.load(open(f, 'rb'))['df'] for f in model_files],
                    axis=0
                )

                for param_name, param_val in vary_param_dict.items():
                    val = vary_param_vals[param_name][param_val]
                    if vary_type[param_name] == "dgp":
                        if np.isscalar(val):
                            results.insert(0, param_name, val)
                        else:
                            results.insert(0, param_name, [val for i in range(results.shape[0])])
                        results.insert(1, param_name + "_name", param_val)
                    elif vary_type[param_name] == "est":
                        results.insert(0, param_name + "_name", copy.deepcopy(results[param_name]))
                results.insert(0, 'rep', i)
                results_list.append(results)
    else:
        for val_name, val in vary_param_vals.items():
            for i in range(args.nreps):
                all_files = glob.glob(oj(path, val_name, 'rep' + str(i), '*'))
                model_files = sorted([f for f in all_files if os.path.basename(f) in model_comparison_files_all])

                if len(model_files) == 0:
                    print('No files found at ', oj(path, val_name, 'rep' + str(i)))
                    continue

                results = pd.concat(
                    [pkl.load(open(f, 'rb'))['df'] for f in model_files],
                    axis=0
                )
                if vary_type == "dgp":
                    if np.isscalar(val):
                        results.insert(0, vary_param_name, val)
                    else:
                        results.insert(0, vary_param_name, [val for i in range(results.shape[0])])
                    results.insert(1, vary_param_name + "_name", val_name)
                    results.insert(2, 'rep', i)
                elif vary_type == "est":
                    results.insert(0, vary_param_name + "_name", copy.deepcopy(results[vary_param_name]))
                    results.insert(1, 'rep', i)
                results_list.append(results)
    
    results_merged = pd.concat(results_list, axis=0)
    pkl.dump(results_merged, open(oj(path, 'results.pkl'), 'wb'))
    results_df = reformat_results(results_merged)
    results_df.to_csv(oj(path, 'results.csv'), index=False)

    print('merged and saved all experiment results successfully!')

# %%
