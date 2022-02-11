import argparse
import os
import pickle as pkl
import time
import warnings
from collections import defaultdict
from os.path import join as oj
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, \
    RandomForestRegressor, BaggingRegressor, BaggingClassifier
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tqdm import tqdm

import config
import util
from util_metrics import tp, fp, neg, pos, specificity_score, pr_auc_score
from imodels.util.data_util import get_clean_dataset
from util import ModelConfig, FIModelConfig, get_rejected_features
import itertools

warnings.filterwarnings("ignore", message="Bins whose width")


def compare_estimators(estimators: List[ModelConfig],
                       fi_estimators: List[FIModelConfig],
                       datasets: List[Tuple],
                       metrics: List[Tuple[str, Callable]],
                       args, ) -> Tuple[dict, dict]:
    """Calculates results given estimators, feature importance estimators, datasets, and metrics.
    Called in run_comparison
    """
    if type(estimators) != list:
        raise Exception("First argument needs to be a list of Models")
    if type(metrics) != list:
        raise Exception("Argument metrics needs to be a list containing ('name', callable) pairs")

    # initialize results with metadata
    results = defaultdict(lambda: [])
    for e in estimators:
        kwargs: dict = e.kwargs  # dict
        for k in kwargs:
            results[k].append(kwargs[k])
    rules = results.copy()

    sklearn_baselines = {
        # insert models here
        RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier,
        RandomForestRegressor, GradientBoostingRegressor, DecisionTreeRegressor,
        BaggingClassifier, BaggingRegressor, GridSearchCV, LogisticRegressionCV, RidgeCV
    }

    # loop over datasets
    for d in datasets:
        if args.verbose:
            print("\tdataset", d[0], 'ests', estimators, 'fi', fi_estimators)
        X, y, feat_names = get_clean_dataset(d[1], data_source=d[2])

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
                if type(est) in sklearn_baselines:
                    est.fit(X_train, y_train)
                else:
                    est.fit(X_train, y_train, feature_names=feat_names)
                end = time.time()

                # things for saving
                rules[d[0]].append(vars(est))

                # loop over fi estimators
                for fi_est in fi_ests:
                    metric_results = {
                        'dataset': d[0],
                        'model': model.name,
                        'fi': fi_est.name,
                        'splitting_strategy': splitting_strategy
                    }
                    fi_score = fi_est.cls(X_test, y_test, est)
                    metric_results['fi_scores'] = fi_score
                    reject_features = None
                    if fi_est.pval:
                        reject_features = get_rejected_features(fi_score, args.alpha)
                    metric_results['est_support'] = reject_features
                    metric_results['complexity'] = util.get_complexity(est)
                    metric_results['time'] = end - start

                    for met_name, met_val in metric_results.items():
                        results[met_name].append(met_val)
    return results, rules


def run_comparison(path: str,
                   datasets: List[Tuple],
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

    results, rules = compare_estimators(estimators=estimators,
                                        fi_estimators=fi_estimators,
                                        datasets=datasets,
                                        metrics=metrics,
                                        args=args)

    estimators_list = [e.name for e in estimators]
    fi_estimators_list = [f.name for f in itertools.chain(*fi_estimators)]
    metrics_list = [m[0] for m in metrics]

    df = pd.DataFrame.from_dict(results)
    df['split_seed'] = args.split_seed

    df_rules = pd.DataFrame.from_dict(rules)
    df_rules['split_seed'] = args.split_seed
    df_rules['estimator'] = estimators_list

    output_dict = {
        # metadata
        'estimators': estimators_list,
        'fi_estimators': fi_estimators_list,
        'comparison_datasets': datasets,
        'metrics': metrics_list,

        # actual values
        'df': df,
        'df_rules': df_rules,
    }
    pkl.dump(output_dict, open(model_comparison_file, 'wb'))


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

    parser.add_argument('--classification_or_regression', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)  # , default='c4')
    parser.add_argument('--fi_model', type=str, default=None)  # , default='c4')
    parser.add_argument('--dataset', type=str, default=None)  # default='reci')
    parser.add_argument('--config', type=str, default='nonlinear_significance')
    parser.add_argument('--alpha', type=float, default=0.05)

    # for multiple reruns, should support varying split_seed
    parser.add_argument('--ignore_cache', action='store_true', default=False)
    parser.add_argument('--splitting_strategy', type=str, default="importances")
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--parallel_id', nargs='+', type=int, default=None)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--regression', action='store_true',
                        help='whether to use regression (sets classification_or_regression)')
    parser.add_argument('--classification', action='store_true',
                        help='whether to use classification (sets classification_or_regression)')
    # parser.add_argument('--ensemble', action='store_true', default=False)
    parser.add_argument('--results_path', type=str,
                        default=oj(os.path.dirname(os.path.realpath(__file__)), 'results'))
    args = parser.parse_args()

    DATASETS_CLASSIFICATION, DATASETS_REGRESSION, \
        ESTIMATORS_CLASSIFICATION, ESTIMATORS_REGRESSION, \
        FI_ESTIMATORS_CLASSIFICATION, FI_ESTIMATORS_REGRESSION = config.get_fi_configs(args.config)

    if args.classification:
        args.classification_or_regression = 'classification'
    elif args.regression:
        args.classification_or_regression = 'regression'
    if args.classification_or_regression is None:
        if args.dataset in [d[0] for d in DATASETS_CLASSIFICATION]:
            args.classification_or_regression = 'classification'
        elif args.dataset in [d[0] for d in DATASETS_REGRESSION]:
            args.classification_or_regression = 'regression'
        else:
            raise ValueError('Either args.classification_or_regression or args.dataset must be properly set!')

    # basic setup
    if args.classification_or_regression == 'classification':
        datasets = DATASETS_CLASSIFICATION
        ests = ESTIMATORS_CLASSIFICATION
        fi_ests = FI_ESTIMATORS_CLASSIFICATION
    elif args.classification_or_regression == 'regression':
        datasets = DATASETS_REGRESSION
        ests = ESTIMATORS_REGRESSION
        fi_ests = FI_ESTIMATORS_REGRESSION

    metrics = get_metrics()

    # filter based on args
    if args.dataset:
        datasets = list(filter(lambda x: args.dataset.lower() == x[0].lower(), datasets))  # strict
        # datasets = list(filter(lambda x: args.dataset.lower() in x[0].lower(), datasets)) # flexible
    if args.model:
        #         ests = list(filter(lambda x: args.model.lower() in x[0].name.lower(), ests))
        ests = list(filter(lambda x: args.model.lower() == x[0].name.lower(), ests))
    if args.fi_model:
        fi_ests = list(filter(lambda x: args.fi_model.lower() == x[0].name.lower(), fi_ests))

    """
    if args.ensemble:
        ests = get_ensembles_for_dataset(args.dataset, test=args.test)
    else:
        ests = get_estimators_for_dataset(args.dataset, test=args.test)
    """

    if len(ests) == 0:
        raise ValueError('No valid estimators', 'dset', args.dataset, 'models', args.model, 'fi', args.fi_model)
    if len(fi_ests) == 0:
        raise ValueError('No valid FI estimators', 'dset', args.dataset, 'models', args.model, 'fi', args.fi_model)
    if len(datasets) == 0:
        raise ValueError('No valid datasets!')
    if args.verbose:
        print('running',
              'datasets', [d[0] for d in datasets],
              'ests', ests,
              'fi_ests', fi_ests)
        print('\tsaving to', args.results_path)
#         print('\tinput arguments:', args.dataset, [d[0] for d in DATASETS_CLASSIFICATION])

    for dataset in tqdm(datasets):
        path = util.get_results_path_from_args(args, dataset[0])
        for est in ests:
            np.random.seed(1)
            run_comparison(path=path,
                           datasets=[dataset],
                           metrics=metrics,
                           estimators=est,
                           fi_estimators=fi_ests,
                           args=args)
    print('completed all experiments successfully!')

#%%
