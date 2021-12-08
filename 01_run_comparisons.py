import argparse
import numpy as np
import os
import pandas as pd
import pickle as pkl
import time
import warnings
from collections import OrderedDict, defaultdict
from os.path import join as oj

from imodels.discretization import ExtraBasicDiscretizer
from imodels.util import data_util
from sklearn import metrics, model_selection
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tqdm import tqdm
from typing import Sequence, Callable

import config
import util
import validate

warnings.filterwarnings("ignore", message="Bins whose width")


def compute_metrics(metrics: OrderedDict[str, Callable],
                    estimator: BaseEstimator,
                    suffix: str,
                    est_time: int,
                    X_eval,
                    y_eval):
    results = {}
    y_pred = estimator.predict(X_eval)
    if args.classification_or_regression == 'classification':
        y_pred_proba = estimator.predict_proba(X_eval)[:, 1]

    for met_name, met in metrics.items():
        if met is not None:
            if args.classification_or_regression == 'regression' \
                    or met_name in ['accuracy', 'f1', 'precision', 'recall']:
                results[met_name + suffix] = met(y_eval, y_pred)
            else:
                results[met_name + suffix] = met(y_eval, y_pred_proba)

        if suffix != '_test':
            results['vars' + suffix] = (vars(estimator))
            results['complexity' + suffix] = util.get_complexity(estimator)
            results['time' + suffix] = est_time

    return results


def compare_estimators(estimators: Sequence[util.Model],
                       dataset: util.Dataset,
                       metrics: OrderedDict[str, Callable],
                       args) -> dict:
    """Runs models on dataset and returns evaluation metrics,
    learned rules, and fitted estimator attributes in dictionaries.
    """
    if type(estimators) != list:
        raise ValueError("First argument needs to be a list of Models")
    if type(metrics) != OrderedDict:
        raise ValueError("Argument metrics needs to be an OrderedDict[str, Callable]")
    if type(dataset) != util.Dataset:
        raise ValueError("Argument dataset needs to be a util.Dataset object")

    # Initialize results dicts with estimator params as index
    est_metrics = defaultdict(lambda: [])
    for e in estimators:
        est_metrics[e.vary_param].append(e.vary_param_val)
        if e.curve_id:
            est_metrics['other_kwargs'].append(e.fixed_kwargs)
            est_metrics['curve_id'].append(e.curve_id)
        else:
            est_metrics['curve_id'].append('default')
    # est_attributes = est_metrics.copy()

    if args.verbose:
        print("\tdataset", dataset.name, 'ests', estimators)
    X, y, feature_names = data_util.get_clean_dataset(dataset.id, data_source=dataset.source)

    # For stablerules we discretize to facilitate rule matching
    if args.config == 'stablerules':
        disc_column_names = np.array(feature_names)[dataset.disc_columns]
        eb_discretizer = ExtraBasicDiscretizer(
            dcols=disc_column_names, n_bins=8, strategy='uniform')
        disc_df = eb_discretizer.fit_transform(pd.DataFrame(X, columns=feature_names))
        X, feature_names = disc_df.values.astype(int), disc_df.columns.values

    # Separate test set before doing any validation
    if args.splitting_strategy in {'train-test-lowdata', 'train-tune-test-lowdata'}:
        test_size = 0.9
    else:
        test_size = 0.2

    if dataset.name == 'csi':
        shuffle = False
        test_size = 744
    else:
        shuffle = True

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size, random_state=args.split_seed, shuffle=shuffle)

    # Sklearn estimators require a slightly different .fit call
    sklearn_baselines = {
        RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier,
        RandomForestRegressor, GradientBoostingRegressor, DecisionTreeRegressor}

    def fit_estimator(estimator: BaseEstimator, X, y) -> tuple[BaseEstimator, float]:
        start = time.time()
        if type(estimator) in sklearn_baselines:
            estimator.fit(X, y)
        else:
            estimator.fit(X, y, feature_names=feature_names)
        end = time.time()
        return estimator, end - start

    # Loop over estimators, recording metrics and fitted attributes
    for model in tqdm(estimators, leave=False):
        # print('kwargs', model.kwargs)
        est = model.cls(**model.kwargs)
        # print(est.criterion)

        all_metric_results = {}
        if args.splitting_strategy in {'cv', 'cv-lowdata'}:

            kf = model_selection.KFold(10, random_state=0, shuffle=True)
            for i, (train_idx, tune_idx) in enumerate(kf.split(X_train)):
                X_train_curr = X_train[train_idx]
                y_train_curr = y_train[train_idx]
                X_tune_curr = X_train[tune_idx]
                y_tune_curr = y_train[tune_idx]

                est, est_time = fit_estimator(est, X_train_curr, y_train_curr)

                # Save metrics and attributes
                suffix = f'_fold_{i}'
                metric_results = compute_metrics(metrics=metrics,
                                                 estimator=est,
                                                 suffix=suffix,
                                                 est_time=est_time,
                                                 X_eval=X_tune_curr,
                                                 y_eval=y_tune_curr)
                all_metric_results = {**all_metric_results, **metric_results}

        if args.splitting_strategy in {'train-tune-test', 'train-tune-test-lowdata'}:
            X_train_curr, X_tune, y_train_curr, y_tune = model_selection.train_test_split(
                X_train, y_train, test_size=0.2, random_state=args.split_seed)

            est, est_time = fit_estimator(est, X_train_curr, y_train_curr)

            suffix = '_tune'
            # est_attributes['vars' + suffix].append(vars(est))
            metric_results = compute_metrics(metrics, est, suffix, est_time, X_tune, y_tune)
            all_metric_results = {**all_metric_results, **metric_results}

        # Always record training and test accuracy, regardless of splitting strategy
        est, est_time = fit_estimator(est, X_train, y_train)

        for suffix, (X_, y_) in zip(['_train', '_test'], [(X_train, y_train), (X_test, y_test)]):
            metric_results = compute_metrics(metrics, est, suffix, est_time, X_, y_)
            all_metric_results = {**all_metric_results, **metric_results}
        # est_attributes['_train'].append(vars(est))

        for met_name, met_val in all_metric_results.items():
            colname = dataset.name + '_' + met_name
            est_metrics[colname].append(met_val)

    return est_metrics  #, est_attributes


def run_comparison(path: str,
                   dataset: util.Dataset,
                   metrics: Sequence[dict[str, Callable]],
                   estimators: Sequence[util.Model],
                   args):

    estimator_name = estimators[0].name
    model_comparison_file = oj(path, f'{estimator_name}_comparisons.pkl')
    if args.parallel_id is not None:
        model_comparison_file = f'_{args.parallel_id[0]}.'.join(model_comparison_file.split('.'))

    if os.path.isfile(model_comparison_file) and not args.ignore_cache:
        print(f'{estimator_name} results already computed. use --ignore_cache to recompute')
        return

    results = compare_estimators(
        estimators=estimators, dataset=dataset, metrics=metrics, args=args)

    estimators_list = [e.name for e in estimators]
    metrics_list = list(metrics.keys())
    df = pd.DataFrame.from_dict(results)
    df.index = estimators_list
    # rule_df = pd.DataFrame.from_dict(rules)
    # rule_df.index = estimators_list

    if args.splitting_strategy in {'cv', 'cv-lowdata'}:
        # Average metrics over the cv folds
        for met_name in metrics:
            in_col_prefix = f'{met_name}_fold'
            out_col = f'{met_name}_cv_mean'
            met_df = df.loc[:, [in_col_prefix in col for col in df.columns]]
            df[out_col] = met_df.mean(axis=1)

    # if args.parallel_id is None:
    #     try:
    #         meta_auc_df = compute_meta_auc(df)
    #     except ValueError as e:
    #         warnings.warn(f'bad complexity range')
    #         meta_auc_df = None

    # meta_auc_df = pd.DataFrame([])
    # if parallel_id is None:
    #     for curr_df, prefix in level_dfs:
    #         try:
    #             curr_meta_auc_df = compute_meta_auc(curr_df, prefix)
    #             meta_auc_df = pd.concat((meta_auc_df, curr_meta_auc_df), axis=1)
    #         except ValueError as e:
    #             warnings.warn(f'bad complexity range for {prefix} datasets')

    output_dict = {
        'estimators': np.unique(estimators_list),
        'dataset': dataset,
        'metrics': metrics_list,
        'df': df,
        # 'rule_df': rule_df,
    }
    # if args.parallel_id is None:
    #     output_dict['meta_auc_df'] = meta_auc_df

    pkl.dump(output_dict, open(model_comparison_file, 'wb'))


def get_metrics(classification_or_regression: str = 'classification') -> OrderedDict:
    mutual = {'complexity': None, 'time': None}
    if classification_or_regression == 'classification':
        return OrderedDict({
            'rocauc': metrics.roc_auc_score,
            'accuracy': metrics.accuracy_score,
            'f1': metrics.f1_score,
            'recall': metrics.recall_score,
            'precision': metrics.precision_score,
            'avg_precision': metrics.average_precision_score,
            'best_accuracy': validate.get_best_accuracy,
            'best_spec_0.9_sens': validate.make_best_spec_high_sens_scorer(0.9),
            'best_spec_0.95_sens': validate.make_best_spec_high_sens_scorer(0.95),
            'best_spec_0.98_sens': validate.make_best_spec_high_sens_scorer(0.98),
            **mutual})
    elif classification_or_regression == 'regression':
        return OrderedDict({
            'r2': metrics.r2_score,
            'explained_variance': metrics.explained_variance_score,
            'neg_mean_squared_error': metrics.mean_squared_error,
            **mutual})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # often-changing args
    parser.add_argument('--classification_or_regression', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)  # , default='c4')
    parser.add_argument('--dataset', type=str, default=None)  # default='reci')
    parser.add_argument('--config', type=str, default='shrinkage')

    # for multiple reruns, should support varying split_seed
    parser.add_argument('--ignore_cache', action='store_true', default=False)
    parser.add_argument('--splitting_strategy', type=str, default="train-test")
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--parallel_id', nargs='+', type=int, default=None)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--regression', action='store_true',
                        help='whether to use regression (sets classification_or_regression)')
    parser.add_argument('--classification', action='store_true',
                        help='whether to use classification (sets classification_or_regression)')
    parser.add_argument('--ensemble', action='store_true', default=False)
    parser.add_argument('--results_path', type=str,
                        default=oj(os.path.dirname(os.path.realpath(__file__)), 'results'))
    args = parser.parse_args()

    assert args.splitting_strategy in {
        'train-test', 'train-tune-test', 'train-test-lowdata', 'train-tune-test-lowdata', 'cv', 'cv-lowdata'}

    DATASETS_CLASSIFICATION, DATASETS_REGRESSION, \
        ESTIMATORS_CLASSIFICATION, ESTIMATORS_REGRESSION = config.get_configs(args.config)

    print('dset', args.dataset, [d.name for d in DATASETS_CLASSIFICATION])
    if args.classification:
        args.classification_or_regression = 'classification'
    elif args.regression:
        args.classification_or_regression = 'regression'
    if args.classification_or_regression is None:
        if args.dataset in [d.name for d in DATASETS_CLASSIFICATION]:
            args.classification_or_regression = 'classification'
        elif args.dataset in [d.name for d in DATASETS_REGRESSION]:
            args.classification_or_regression = 'regression'
        else:
            raise ValueError(
                'Either args.classification_or_regression or args.dataset must be properly set!')

    # basic setup
    if args.classification_or_regression == 'classification':
        datasets = DATASETS_CLASSIFICATION
        estimator_lists = ESTIMATORS_CLASSIFICATION
    elif args.classification_or_regression == 'regression':
        datasets = DATASETS_REGRESSION
        estimator_lists = ESTIMATORS_REGRESSION

    metric_list = get_metrics(args.classification_or_regression)

    # filter based on args
    if args.dataset:
        datasets = [dset for dset in datasets if dset.name == args.dataset]  # strict
        # datasets = list(filter(lambda x: args.dataset.lower() in x[0].lower(), datasets)) # flexible
    if args.model:
        #         ests = list(filter(lambda x: args.model.lower() in x[0].name.lower(), ests))
        estimator_lists = [
            est_list for est_list in estimator_lists if args.model == est_list[0].name]

    # if args.ensemble:
    #     ests = get_ensembles_for_dataset(args.dataset, test=args.test)
    # else:
    #     ests = get_estimators_for_dataset(args.dataset, test=args.test)

    if len(estimator_lists) == 0:
        raise ValueError('No valid estimators', 'dset', args.dataset, 'models', args.model)
    if len(datasets) == 0:
        raise ValueError('No valid datasets!')
    if args.verbose:
        print('running',
              'datasets', [d.name for d in datasets],
              'estimators', estimator_lists)
        print('saving to', args.results_path)

    for dataset in tqdm(datasets):
        path = util.get_results_path_from_args(args, dataset.name)
        for est_list in estimator_lists:
            np.random.seed(1)
            run_comparison(path=path,
                           dataset=dataset,
                           metrics=metric_list,
                           estimators=est_list,
                           args=args)
    print('completed all experiments successfully!')
