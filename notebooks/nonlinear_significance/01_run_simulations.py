# Example usage: run in command line
# cd notebooks/nonlinear_significance
# python 01_run_simulations.py --nreps 2 --config test --split_seed 331 --ignore_cache
# python 01_run_simulations.py --nreps 2 --config test --split_seed 331 --ignore_cache --create_rmd

import copy
import os
from os.path import join as oj
import glob
import argparse
import pickle as pkl
import time
import warnings
from scipy import stats

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
                      if fi_estimator.model_type in est_type]
        if len(fi_ests_ls) == 0:
            continue

        # get kwargs for all fi_ests
        fi_kwargs = {}
        for fi_est in fi_ests_ls:
            fi_kwargs.update(fi_est.kwargs)

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
                fi_score = fi_est.cls(X_test, y_test, est, **fi_est.kwargs)
                metric_results['fi_scores'] = copy.deepcopy(fi_score)
                reject_features = None
                if fi_est.pval:
                    if 'rejections' in fi_score.columns:
                        reject_features = copy.deepcopy(fi_score[['var', 'rejections']]).\
                            rename(columns={'rejections':'importance'})
                    else:
                        reject_features = get_rejected_features(fi_score, args.alpha)
                        fi_score['importance'] = -fi_score['importance']  # because lower p-value is more significant
                metric_results['est_support'] = reject_features
                if np.max(support) != np.min(support):
                    for i, (met_name, met) in enumerate(metrics):
                        if met is not None:
                            if met_name in ['f1', 'precision', 'recall', 'specificity', 'fp', 'tp', 'neg', 'pos']:
                                if fi_est.pval:
                                    metric_results[met_name] = met(support, reject_features['importance'])
                                else:
                                    metric_results[met_name] = None
                            else:
                                metric_results[met_name] = met(support, fi_score['importance'])
                    if args.r2:
                        metric_results['r2_rocauc'] = None
                        metric_results['r2_prauc'] = None
                        metric_results['r2_pval_cor'] = None
                        if 'r2' in fi_score.columns:
                            metric_results['r2_rocauc'] = roc_auc_score(support, fi_score['r2'])
                            metric_results['r2_prauc'] = pr_auc_score(support, fi_score['r2'])
                            metric_results['r2_pval_cor'], _ = stats.spearmanr(fi_score['importance'], fi_score['r2'])

                # metric_results['complexity'] = util.get_complexity(est)
                metric_results['time'] = end - start

                # initialize results with metadata and metric results
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


def reformat_results(results):
    results = results.reset_index().drop(columns=['index'])
    fi_scores = pd.concat(results.pop('fi_scores').to_dict()). \
        reset_index(level=0).rename(columns={'level_0': 'index'})
    if not results['est_support'].isnull().all():
        est_support = pd.concat(results.pop('est_support').to_dict()). \
            reset_index(level=0).rename(columns={'level_0': 'index', 'importance': 'support'})
        joined_df = pd.merge(fi_scores, est_support, how="outer", on=['index', 'var'])
    else:
        joined_df = fi_scores
    results_df = pd.merge(results, joined_df, left_index=True, right_on="index")
    return results_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--nreps', type=int, default=2)
    parser.add_argument('--model', type=str, default=None)  # , default='c4')
    parser.add_argument('--fi_model', type=str, default=None)  # , default='c4')
    parser.add_argument('--config', type=str, default='test')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--r2', action='store_true', default=False)
    parser.add_argument('--omit_vars', type=str, default=None)  # comma-separated string of variables to omit

    # for multiple reruns, should support varying split_seed
    parser.add_argument('--ignore_cache', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--parallel_id', nargs='+', type=int, default=None)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--results_path', type=str,
                        default=oj(os.path.dirname(os.path.realpath(__file__)), 'results'))

    # arguments for rmd output of results
    parser.add_argument('--create_rmd', action='store_true', default=False)
    parser.add_argument('--show_vars', type=int, default=None)

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

    if args.omit_vars is not None:
        results_dir = oj(args.results_path, args.config + "_omitted_vars")
    else:
        results_dir = oj(args.results_path, args.config)
    path = oj(results_dir, "varying_" + vary_param_name, "seed" + str(args.split_seed))
    os.makedirs(path, exist_ok=True)

    eval_out = defaultdict(list)

    vary_type = None
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
            fi_est_kwargs = list(itertools.chain(*[list(fi_est.kwargs.keys()) for fi_est in list(itertools.chain(*fi_ests))]))
            if vary_param_name in est_kwargs:
                vary_type = "est"
            elif vary_param_name in fi_est_kwargs:
                vary_type = "fi_est"
            else:
                raise ValueError('Invalid vary_param_name.')

        for i in tqdm(range(args.nreps)):
            os.makedirs(oj(path, val_name, "rep" + str(i)), exist_ok=True)
            np.random.seed(i)
            X = X_dgp(**X_params_dict)
            y, support, beta = y_dgp(X, **y_params_dict, return_support=True)
            if args.omit_vars is not None:
                omit_vars = np.unique([int(x.strip()) for x in args.omit_vars.split(",")])
                support = np.delete(support, omit_vars)
                X = np.delete(X, omit_vars, axis=1)
                del beta  # note: beta is not currently supported when using omit_vars

            for est in ests:
                results = run_comparison(path=oj(path, val_name, "rep" + str(i)),
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
            all_files = glob.glob(oj(path, val_name, 'rep' + str(i), '*'))
            model_files = sorted([f for f in all_files if '_comparisons' in f])

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
            elif vary_type == "est" or vary_type == "fi_est":
                results.insert(0, vary_param_name + "_name", copy.deepcopy(results[vary_param_name]))
                results.insert(1, 'rep', i)
            results_list.append(results)
    results_merged = pd.concat(results_list, axis=0)
    pkl.dump(results_merged, open(oj(path, 'results.pkl'), 'wb'))
    results_df = reformat_results(results_merged)
    results_df.to_csv(oj(path, 'results.csv'), index=False)

    print('merged and saved all experiment results successfully!')

    if args.create_rmd:
        if args.show_vars is None:
            show_vars = 'NULL'
        else:
            show_vars = args.show_vars
        os.system(
            'Rscript -e "rmarkdown::render(\'{}\', params = list(results_dir = \'{}\', vary_param_name = \'{}\', seed = {}, keep_vars = {}), output_file = \'{}\', quiet = TRUE)"'.format(
                "02_simulation_results.Rmd",
                results_dir, vary_param_name, str(args.split_seed), str(show_vars),
                oj(path, "simulation_results.html"))
        )
        print("created rmd of simulation results successfully!")

# %%
