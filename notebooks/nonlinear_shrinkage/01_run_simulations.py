import os
from os.path import join as oj
import glob
import argparse
import pickle as pkl
import time
import warnings


from tqdm import tqdm
import sys
sys.path.append('.')
from simulations_util import *

sys.path.append('..')
import sim_config
sys.path.append('../..')
sys.path.append('../../..')

from collections import defaultdict
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

import util
from util_metrics import tp, fp, neg, pos, specificity_score, pr_auc_score
from util import ModelConfig, FIModelConfig, get_rejected_features
import itertools

warnings.filterwarnings("ignore", message="Bins whose width")

from viz import *


def compare_estimators(estimators: List[ModelConfig],
                       fi_estimators: List[FIModelConfig],
                       X, y, support: List,
                       metrics: List[Tuple[str, Callable]],
                       args, ) -> Tuple[dict, dict]:
    """Calculates results given estimators, feature importance estimators, datasets, and metrics.
    Called in run_comparison
    """
    if type(estimators) != list:
        raise Exception("First argument needs to be a list of Models")
    if type(metrics) != list:
        raise Exception("Argument metrics needs to be a list containing ('name', callable) pairs")

    # initialize results
    results = defaultdict(lambda: [])

    # loop over model estimators
    for model in tqdm(estimators, leave=False):
        est = model.cls(**model.kwargs)
        est_type = model.model_type
        fi_ests_ls = [fi_estimator for fi_estimator in itertools.chain(*fi_estimators) \
                      if fi_estimator.model_type == est_type]
        if len(fi_ests_ls) == 0:
            continue

        # get groups of estimators for each splitting strategy
        fi_ests_dict = defaultdict(list)
        for fi_est in fi_ests_ls:
            fi_ests_dict[fi_est.splitting_strategy].append(fi_est)

        # loop over splitting strategies
        for splitting_strategy, fi_ests in fi_ests_dict.items():
            # implement provided splitting strategy
            if splitting_strategy is not None:
                X_train, X_tune, X_test, y_train, y_tune, y_test = (
                    util.apply_splitting_strategy(X, y, splitting_strategy, args.split_seed))
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

            # loop over fi estimators
            for fi_est in fi_ests:
                metric_results = {
                    'model': model.name,
                    'fi': fi_est.name,
                    'splitting_strategy': splitting_strategy
                }
                fi_score = fi_est.cls(X_test, y_test, est)
                metric_results['fi_scores'] = fi_score
                reject_features = None
                if fi_est.pval:
                    fi_score['importance'] = -fi_score['importance']
                    reject_features = get_rejected_features(fi_score, args.alpha)
                metric_results['est_support'] = reject_features
                for i, (met_name, met) in enumerate(metrics):
                    if met is not None:
                        if met_name in ['f1', 'precision', 'recall', 'specificity', 'fp', 'tp', 'neg', 'pos']:
                            if fi_est.pval:
                                metric_results[met_name] = met(support, reject_features)
                            else:
                                metric_results[met_name] = None
                        else:
                            metric_results[met_name] = met(support, fi_score['importance'])
                metric_results['complexity'] = util.get_complexity(est)
                metric_results['time'] = end - start

                # initialize results with metadata and metric results
                kwargs: dict = model.kwargs  # dict
                for k in kwargs:
                    results[k].append(kwargs[k])
                for met_name, met_val in metric_results.items():
                    results[met_name].append(met_val)
    return results


def run_comparison(path: str,
                   X, y, support: List,
                   metrics: List[Tuple[str, Callable]],
                   estimators: List[ModelConfig],
                   fi_estimators: List[FIModelConfig],
                   args):
    estimator_name = estimators[0].name.split(' - ')[0]
    model_comparison_file = oj(path, f'{estimator_name}_comparisons.pkl')
    if args.parallel_id is not None:
        model_comparison_file = f'_{args.parallel_id[0]}.'.join(model_comparison_file.split('.'))

    if os.path.isfile(model_comparison_file) and not args.ignore_cache:
        print(f'{estimator_name} results already computed and cached. use --ignore_cache to recompute')
        return

    results = compare_estimators(estimators=estimators,
                                 fi_estimators=fi_estimators,
                                 X=X, y=y, support=support,
                                 metrics=metrics,
                                 args=args)

    estimators_list = [e.name for e in estimators]
    fi_estimators_list = [f.name for f in itertools.chain(*fi_estimators)]
    metrics_list = [m[0] for m in metrics]

    df = pd.DataFrame.from_dict(results)
    df['split_seed'] = args.split_seed

    output_dict = {
        # metadata
        'sim_name': args.config,
        'estimators': estimators_list,
        'fi_estimators': fi_estimators_list,
        'metrics': metrics_list,

        # actual values
        'df': df,
    }
    pkl.dump(output_dict, open(model_comparison_file, 'wb'))
    return df


def get_metrics():
    mutual = [('complexity', None), ('time', None)]
    return [
               ('rocauc', roc_auc_score),
               ('prauc', pr_auc_score),
               ('f1', f1_score),
               ('precision', precision_score),
               ('recall', recall_score),
               ('specificity', specificity_score),
               ('fp', fp),
               ('tp', tp),
               ('neg', neg),
               ('pos', pos)
           ] + mutual



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--nreps', type=int, default=2)
    parser.add_argument('--model', type=str, default=None)  # , default='c4')
    parser.add_argument('--fi_model', type=str, default=None)  # , default='c4')
    parser.add_argument('--config', type=str, default='normal_linear_dgp')
    parser.add_argument('--alpha', type=float, default=0.05)

    # for multiple reruns, should support varying split_seed
    parser.add_argument('--ignore_cache', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--parallel_id', nargs='+', type=int, default=None)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--results_path', type=str,
                        default=oj(os.path.dirname(os.path.realpath(__file__)), 'results'))
    args = parser.parse_args()

    ests, fi_ests, \
    X_dgp, X_params_dict, y_dgp, y_params_dict, \
    vary_param_name, vary_param_vals = sim_config.get_fi_sims_configs(args.config)

    metrics = get_metrics()

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

    path = oj(args.results_path, args.config,
              "varying_" + vary_param_name, "seed" + str(args.split_seed))
    os.makedirs(path, exist_ok=True)

    eval_out = defaultdict(list)

    for val_name, val in vary_param_vals.items():
        print(vary_param_name, ":", val_name)
        if vary_param_name in X_params_dict.keys() and vary_param_name in y_params_dict.keys():
            raise ValueError('Cannot vary over parameter in both X and y DGPs.')
        elif vary_param_name in X_params_dict.keys():
            X_params_dict[vary_param_name] = val
        elif vary_param_name in y_params_dict.keys():
            y_params_dict[vary_param_name] = val
        else:
            raise ValueError('Invalid vary_param_name.')

        for i in tqdm(range(args.nreps)):
            os.makedirs(oj(path, val_name, "rep"+str(i)), exist_ok=True)
            np.random.seed(i)
            X = X_dgp(**X_params_dict)
            y, support, beta = y_dgp(X, **y_params_dict, return_support=True)

            for est in ests:
                results = run_comparison(path=oj(path, val_name, "rep"+str(i)),
                                         X=X, y=y, support=support,
                                         metrics=metrics,
                                         estimators=est,
                                         fi_estimators=fi_ests,
                                         args=args)

    print('completed all experiments successfully!')

    # aggregate results
    results_list = []
    for val_name, val in vary_param_vals.items():
        for i in range(args.nreps):
            all_files = glob.glob(oj(path, val_name, 'rep'+str(i), '*'))
            model_files = sorted([f for f in all_files if '_comparisons' in f])

            if len(model_files) == 0:
                print('No files found at ', oj(path, val_name, 'rep'+str(i)))
                continue

            results = pd.concat(
                [pkl.load(open(f, 'rb'))['df'] for f in model_files],
                axis=0
            )
            results.insert(0, vary_param_name, val)
            results.insert(1, vary_param_name + "_name", val_name)
            results.insert(2, 'rep', i)
            results_list.append(results)
    results_merged = pd.concat(results_list, axis=0)
    pkl.dump(results_merged, open(oj(path, 'results.pkl'), 'wb'))

    print('merged and saved all experiment results successfully!')


