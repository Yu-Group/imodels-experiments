# Example usage: run in command line
# cd feature_importance/
# python 00_run_prediction_and_importance.py --nreps 2 --config test --split_seed 12345
# python 00_run_prediction_and_importance.py --nreps 2 --config test --split_seed 12345 --ignore_cache

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
from util import ModelConfig, FIModelConfig, apply_splitting_strategy, auroc_score, auprc_score

from sklearn.metrics import accuracy_score, f1_score, recall_score, \
    precision_score, average_precision_score, r2_score, explained_variance_score, \
    mean_squared_error, mean_absolute_error, log_loss
from sksurv.metrics import brier_score, concordance_index_censored, concordance_index_ipcw

warnings.filterwarnings("ignore", message="Bins whose width")


def compare_estimators(estimators: List[ModelConfig],
                       fi_estimators: List[FIModelConfig],
                       X, y, support: List,
                       metrics: List[Tuple[str, Callable]],
                       fi_metrics: List[Tuple[str, Callable]],
                       args, rep) -> Tuple[dict, dict]:
    """Calculates results given estimators, feature importance estimators, and datasets.
    Called in run_comparison
    """
    if type(estimators) != list:
        raise Exception("First argument needs to be a list of Models")
    if type(metrics) != list:
        raise Exception("Argument metrics needs to be a list containing ('name', callable) pairs")
    if type(fi_metrics) != list:
        raise Exception("Argument fi_metrics needs to be a list containing ('name', callable) pairs")

    # initialize results
    pred_results = defaultdict(lambda: [])
    fi_results = defaultdict(lambda: [])

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
                X_train, X_tune, X_test, y_train, y_tune, y_test = apply_splitting_strategy(
                    X, y, splitting_strategy, args.split_seed + rep
                )
            else:
                X_train = X
                X_tune = X
                X_test = X
                y_train = y
                y_tune = y
                y_test = y

            # fit model
            start = time.time()
            est.fit(X_train, y_train)
            end = time.time()

            # prediction results
            pred_metric_results = {
                'model': model.name,
                'splitting_strategy': splitting_strategy
            }
            y_pred = est.predict(X_test)
            if args.mode in ['binary_classification', 'multiclass_classification']:
                y_pred_proba = est.predict_proba(X_test)
                if args.mode == 'binary_classification':
                    y_pred_proba = y_pred_proba[:, 1]
            else:
                y_pred_proba = y_pred
            for met_name, met in metrics:
                if met is not None:
                    if met_name in ["rocauc", "prauc", "logloss", "avg_precision"]:
                        pred_metric_results[met_name] = met(y_test, y_pred_proba)
                    elif args.mode != "survival":
                        pred_metric_results[met_name] = met(y_test, y_pred)
                    elif args.mode == "survival":
                        if met_name == "concordance_index_censored":
                            pred_metric_results[met_name] = met(y_test['censor'], y_test['time'], y_pred)[0]
                        else:
                            pred_metric_results[met_name] = met(y_train, y_test, y_pred)[0]

            pred_metric_results['predictions'] = copy.deepcopy(pd.DataFrame(y_pred_proba))
            pred_metric_results['time'] = copy.deepcopy(end - start)

            # initialize results with metadata and metric results
            kwargs: dict = model.kwargs  # dict
            for k in kwargs:
                pred_results[k].append(kwargs[k])
            for met_name, met_val in pred_metric_results.items():
                pred_results[met_name].append(met_val)

            # loop over fi estimators
            for fi_est in fi_ests:
                fi_metric_results = {
                    'model': model.name,
                    'fi': fi_est.name,
                    'splitting_strategy': splitting_strategy
                }
                start = time.time()
                fi_score = fi_est.cls(X_test, y_test, copy.deepcopy(est), **fi_est.kwargs)
                end = time.time()
                if support is None:
                    fi_metric_results['fi_scores'] = copy.deepcopy(fi_score)
                else:
                    support_df = pd.DataFrame({"var": np.arange(len(support)),
                                               "true_support": support})
                    fi_metric_results['fi_scores'] = pd.merge(copy.deepcopy(fi_score), support_df, on="var", how="left")
                    if np.max(support) != np.min(support):
                        for i, (met_name, met) in enumerate(fi_metrics):
                            if met is not None:
                                imp_vals = copy.deepcopy(fi_score["importance"])
                                imp_vals[imp_vals == float("-inf")] = -sys.maxsize - 1
                                imp_vals[imp_vals == float("inf")] = sys.maxsize - 1
                                if fi_est.ascending:
                                    imp_vals[np.isnan(imp_vals)] = -sys.maxsize - 1
                                    fi_metric_results[met_name] = met(support, imp_vals)
                                else:
                                    imp_vals[np.isnan(imp_vals)] = sys.maxsize - 1
                                    fi_metric_results[met_name] = met(support, -imp_vals)
                fi_metric_results['time'] = copy.deepcopy(end - start)

                # initialize results with metadata and metric results
                kwargs: dict = model.kwargs  # dict
                for k in kwargs:
                    fi_results[k].append(kwargs[k])
                for k in fi_kwargs:
                    if k in fi_est.kwargs:
                        fi_results[k].append(str(fi_est.kwargs[k]))
                    else:
                        fi_results[k].append(None)
                for met_name, met_val in fi_metric_results.items():
                    fi_results[met_name].append(met_val)

    return pred_results, fi_results


def run_comparison(rep: int,
                   path: str,
                   X, y,
                   support: List,
                   metrics: List[Tuple[str, Callable]],
                   fi_metrics: List[Tuple[str, Callable]],
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
        return None, None

    pred_results, fi_results = compare_estimators(
        estimators=estimators, fi_estimators=fi_estimators,
        X=X, y=y, support=support,
        metrics=metrics, fi_metrics=fi_metrics,
        args=args, rep=rep
    )

    estimators_list = [e.name for e in estimators]
    metrics_list = [m[0] for m in metrics]
    fi_metrics_list = [m[0] for m in fi_metrics]

    pred_df = pd.DataFrame.from_dict(pred_results)
    pred_df['split_seed'] = args.split_seed + rep

    fi_df = pd.DataFrame.from_dict(fi_results)
    fi_df['split_seed'] = args.split_seed + rep

    if args.nosave_cols is not None:
        nosave_cols = np.unique([x.strip() for x in args.nosave_cols.split(",")])
    else:
        nosave_cols = []
    for col in nosave_cols:
        if col in pred_df.columns:
            pred_df = pred_df.drop(columns=[col])
        if col in fi_df.columns:
            fi_df = fi_df.drop(columns=[col])

    for model_comparison_file, fi_estimator in zip(model_comparison_files, fi_estimators):
        output_dict = {
            # metadata
            'sim_name': args.config,
            'estimators': estimators_list,
            'fi_estimators': fi_estimator.name,
            'metrics': fi_metrics_list,
            # actual values
            'df': fi_df.loc[fi_df.fi == fi_estimator.name],
        }
        pkl.dump(output_dict, open(model_comparison_file, 'wb'))
    pred_output_dict = {
        # metadata
        'sim_name': args.config,
        'estimators': estimators_list,
        'metrics': metrics_list,
        # actual values
        'df': pred_df
    }
    pkl.dump(pred_output_dict, open(oj(path, f'{estimator_name}_comparisons.pkl'), 'wb'))

    return pred_df, fi_df


def reformat_pred_results(results):
    results = results.reset_index().drop(columns=['index'])
    predictions = pd.concat(results.pop('predictions').to_dict()). \
        reset_index(level=[0, 1]).rename(columns={'level_0': 'index', 'level_1': 'sample_id'})
    pred_results_df = pd.merge(results, predictions, left_index=True, right_on="index")
    return results, pred_results_df


def reformat_fi_results(results):
    results = results.reset_index().drop(columns=['index'])
    fi_scores = pd.concat(results.pop('fi_scores').to_dict()). \
        reset_index(level=0).rename(columns={'level_0': 'index'})
    fi_results_df = pd.merge(results, fi_scores, left_index=True, right_on="index")
    return results, fi_results_df


def get_metrics(mode: str = 'regression'):
    assert mode in {'regression', 'binary_classification', 'multiclass_classification', 'survival'}
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
    elif mode == 'survival':
        # TODO: replace with correct metrics
        return [
            #('brier', brier_score),
            ('concordance_index_censored', concordance_index_censored),
            ('concordance_index_ipcw', concordance_index_ipcw),
        ]


def get_fi_metrics():
    return [('rocauc', auroc_score), ('prauc', auprc_score)]


def run_simulation(i, path, val_name, X_params_dict, X_dgp, y_params_dict, y_dgp,
                   ests, fi_ests, metrics, fi_metrics, args):
    if val_name is None:
        output_path = oj(path, "rep" + str(i))
    else:
        output_path = oj(path, val_name, "rep" + str(i))
    os.makedirs(output_path, exist_ok=True)
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
    if args.subsample_n is not None:
        if args.subsample_n < X.shape[0]:
            keep_rows = np.random.choice(X.shape[0], args.subsample_n, replace=False)
            X = X.iloc[keep_rows]
            y = y.iloc[keep_rows]
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pd.DataFrame):
        y = y.to_numpy()
    for est in ests:
        for idx in range(len(est)):
            if "random_state" in est[idx].kwargs.keys():
                est[idx].kwargs["random_state"] = i + 100
        pred_results, fi_results = run_comparison(
            rep=i,
            path=output_path,
            X=X, y=y,
            support=support,
            metrics=metrics,
            fi_metrics=fi_metrics,
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

    parser.add_argument('--nreps', type=int, default=2)
    parser.add_argument('--mode', type=str, default='survival')
    # parser.add_argument('--mode', type=str, default='regression')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--fi_model', type=str, default=None)
    parser.add_argument('--config', type=str, default='mdi_plus_survival.test')
    # parser.add_argument('--config', type=str, default='test')
    parser.add_argument('--subsample_n', type=int, default=None)
    parser.add_argument('--omit_vars', type=str, default=None)  # comma-separated string of variables to omit
    parser.add_argument('--nosave_cols', type=str, default=None)

    # for multiple reruns, should support varying split_seed
    parser.add_argument('--ignore_cache', action='store_true', default=False)
    # parser.add_argument('--splitting_strategy', type=str, default="train-test")
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--parallel_id', nargs='+', type=int, default=None)
    parser.add_argument('--n_cores', type=int, default=None)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--results_path', type=str, default=default_dir)

    args = parser.parse_args()

    # assert args.splitting_strategy in {
    #     'train-test', 'train-tune-test', 'train-test-lowdata', 'train-tune-test-lowdata'
    # }
    assert args.mode in {
        'regression', 'binary_classification', 'multiclass_classification', 'survival'
    }

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
    fi_metrics = get_fi_metrics()

    if args.model:
        ests = list(filter(lambda x: args.model.lower() == x[0].name.lower(), ests))
    if args.fi_model:
        fi_ests = list(filter(lambda x: args.fi_model.lower() == x[0].name.lower(), fi_ests))

    if len(ests) == 0:
        raise ValueError('No valid estimators', 'sim', args.config, 'models', args.model)
    if len(fi_ests) == 0:
        raise ValueError('No valid FI estimators', 'sim', args.config, 'models', args.model, 'fi', args.fi_model)
    if args.verbose:
        print('running', args.config,
              'ests', ests,
              'fi_ests', fi_ests)
        print('\tsaving to', args.results_path)

    if args.omit_vars is not None:
        results_dir = oj(args.results_path, args.config + "_omitted_vars")
    else:
        results_dir = oj(args.results_path, args.config)
    if vary_param_name is None:
        path = oj(results_dir, "seed" + str(args.split_seed))
    elif isinstance(vary_param_name, list):
        path = oj(results_dir, "varying_" + "_".join(vary_param_name), "seed" + str(args.split_seed))
    else:
        path = oj(results_dir, "varying_" + vary_param_name, "seed" + str(args.split_seed))
    os.makedirs(path, exist_ok=True)

    eval_out = defaultdict(list)

    vary_type = None
    if vary_param_name is None:
        if args.parallel:
            futures = [
                dask.delayed(run_simulation)(
                    i, path, None, X_params_dict, X_dgp, y_params_dict, y_dgp,
                    ests, fi_ests, metrics, fi_metrics, args
                ) for i in range(args.nreps)
            ]
            results = dask.compute(*futures)
        else:
            results = [
                run_simulation(
                    i, path, None, X_params_dict, X_dgp, y_params_dict, y_dgp,
                    ests, fi_ests, metrics, fi_metrics, args
                ) for i in range(args.nreps)
            ]
        assert all(results)
    elif isinstance(vary_param_name, list):  # multiple parameters are being varied
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
                    fi_est_kwargs = list(
                        itertools.chain(*[list(fi_est.kwargs.keys()) for fi_est in list(itertools.chain(*fi_ests))]))
                    if param_name in est_kwargs:
                        vary_type[param_name] = "est"
                    elif param_name in fi_est_kwargs:
                        vary_type[param_name] = "fi_est"
                    else:
                        raise ValueError('Invalid vary_param_name.')

            if args.parallel:
                futures = [
                    dask.delayed(run_simulation)(
                        i, path, "_".join(vary_param_dict.values()), X_params_dict, X_dgp,
                        y_params_dict, y_dgp, ests, fi_ests, metrics, fi_metrics, args
                    ) for i in range(args.nreps)
                ]
                results = dask.compute(*futures)
            else:
                results = [
                    run_simulation(
                        i, path, "_".join(vary_param_dict.values()), X_params_dict, X_dgp,
                        y_params_dict, y_dgp, ests, fi_ests, metrics, fi_metrics, args
                    ) for i in range(args.nreps)
                ]
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
                fi_est_kwargs = list(
                    itertools.chain(*[list(fi_est.kwargs.keys()) for fi_est in list(itertools.chain(*fi_ests))]))
                if vary_param_name in est_kwargs:
                    vary_type = "est"
                elif vary_param_name in fi_est_kwargs:
                    vary_type = "fi_est"
                else:
                    raise ValueError('Invalid vary_param_name.')

            if args.parallel:
                futures = [
                    dask.delayed(run_simulation)(
                        i, path, val_name, X_params_dict, X_dgp, y_params_dict, y_dgp,
                        ests, fi_ests, metrics, fi_metrics, args
                    ) for i in range(args.nreps)
                ]
                results = dask.compute(*futures)
            else:
                results = [
                    run_simulation(
                        i, path, val_name, X_params_dict, X_dgp, y_params_dict, y_dgp,
                        ests, fi_ests, metrics, fi_metrics, args
                    ) for i in range(args.nreps)
                ]
            assert all(results)

    print('completed all experiments successfully!')

    # get model file names
    pred_results_files_all = []
    fi_results_files_all = []
    for est in ests:
        estimator_name = est[0].name.split(' - ')[0]
        fi_estimators_all = [fi_estimator for fi_estimator in itertools.chain(*fi_ests) \
                             if fi_estimator.model_type in est[0].model_type]
        fi_results_files_all += [f'{estimator_name}_{fi_estimator.name}_comparisons.pkl'
                                 for fi_estimator in fi_estimators_all]
        pred_results_files_all += [f'{estimator_name}_comparisons.pkl']

    # aggregate results
    results_files = {"pred": pred_results_files_all, "fi": fi_results_files_all}
    for result_type, results_files_all in results_files.items():
        results_list = []
        if vary_param_name is None:
            for i in range(args.nreps):
                all_files = glob.glob(oj(path, 'rep' + str(i), '*'))
                model_files = sorted([f for f in all_files if os.path.basename(f) in results_files_all])
                if len(model_files) == 0:
                    print('No ', result_type, ' files found at ', oj(path, 'rep' + str(i)))
                    continue
                results = pd.concat(
                    [pkl.load(open(f, 'rb'))['df'] for f in model_files], axis=0
                )
                results.insert(0, 'rep', i)
                results_list.append(results)
        elif isinstance(vary_param_name, list):
            for vary_param_dict in vary_param_dicts:
                val_name = "_".join(vary_param_dict.values())
                for i in range(args.nreps):
                    all_files = glob.glob(oj(path, val_name, 'rep' + str(i), '*'))
                    model_files = sorted([f for f in all_files if os.path.basename(f) in results_files_all])
                    if len(model_files) == 0:
                        print('No ', result_type, ' files found at ', oj(path, val_name, 'rep' + str(i)))
                        continue
                    results = pd.concat(
                        [pkl.load(open(f, 'rb'))['df'] for f in model_files], axis=0
                    )
                    for param_name, param_val in vary_param_dict.items():
                        val = vary_param_vals[param_name][param_val]
                        if vary_type[param_name] == "dgp":
                            if np.isscalar(val):
                                results.insert(0, param_name, val)
                            else:
                                results.insert(0, param_name, [val for i in range(results.shape[0])])
                            results.insert(1, param_name + "_name", param_val)
                        elif vary_type[param_name] == "est" or vary_type[param_name] == "fi_est":
                            results.insert(0, param_name + "_name", copy.deepcopy(results[param_name]))
                    results.insert(0, 'rep', i)
                    results_list.append(results)
        else:
            for val_name, val in vary_param_vals.items():
                for i in range(args.nreps):
                    all_files = glob.glob(oj(path, val_name, 'rep' + str(i), '*'))
                    model_files = sorted([f for f in all_files if os.path.basename(f) in results_files_all])
                    if len(model_files) == 0:
                        print('No ', result_type, ' files found at ', oj(path, val_name, 'rep' + str(i)))
                        continue
                    results = pd.concat(
                        [pkl.load(open(f, 'rb'))['df'] for f in model_files], axis=0
                    )
                    if vary_type == "dgp":
                        if np.isscalar(val):
                            results.insert(0, vary_param_name, val)
                        else:
                            results.insert(0, vary_param_name, [val for i in range(results.shape[0])])
                        results.insert(1, vary_param_name + "_name", val_name)
                        results.insert(2, 'rep', i)
                    elif vary_type == "est" or vary_type == "fi_est":
                        results.insert(0, vary_param_name + "_name", copy.deepcopy(results[vary_param_name]))
                        results.insert(1, 'rep', i)
                    results_list.append(results)
        results_merged = pd.concat(results_list, axis=0)
        pkl.dump(results_merged, open(oj(path, f'{result_type}_results.pkl'), 'wb'))
        if result_type == "pred":
            reformat_results = reformat_pred_results
        elif result_type == "fi":
            reformat_results = reformat_fi_results
        results, results_df = reformat_results(results_merged)
        results.to_csv(oj(path, f'{result_type}_results.csv'), index=False)
        results_df.to_csv(oj(path, f'{result_type}_results_full.csv'), index=False)

    print('merged and saved all experiment results successfully!')
# %%
