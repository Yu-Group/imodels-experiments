import argparse
import os
import pickle as pkl
import time
import inspect
import warnings
from collections import defaultdict
from os.path import join as oj
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, recall_score, \
    precision_score, r2_score, explained_variance_score, mean_squared_error
from tqdm import tqdm

import config
import util
from imodels.util.data_util import get_clean_dataset
from imodels.util.tree_interaction_utils import (get_gt, interaction_fpr, interaction_f1,
                                                 interaction_tpr, get_interacting_features, get_important_features)
from util import ModelConfig, get_interaction_score, get_importances
from validate import get_best_accuracy

warnings.filterwarnings("ignore", message="Bins whose width")


def compare_estimators(estimators: List[ModelConfig],
                       dataset: Tuple,
                       metrics: List[Tuple[str, Callable]],
                       args, ) -> Tuple[dict, dict]:
    """Calculates results given estimators, dataset, and metrics.
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

    # scores = results.copy()

    # loop over dataset
    d = dataset
    if args.verbose:
        print("\tdataset", d[0], 'ests', estimators)
    X, y, feat_names = get_clean_dataset(d[1], data_source=d[2])

    # implement provided splitting strategy
    X_train, X_tune, X_test, y_train, y_tune, y_test = (
        util.apply_splitting_strategy(X, y, args.splitting_strategy, args.split_seed))

    # loop over estimators
    for model in tqdm(estimators, leave=False):
        est = model.cls(**model.kwargs)

        start = time.time()
        fit_parameters = inspect.signature(est.fit).parameters.keys()
        if 'feature_names' in fit_parameters:
            est.fit(X_train, y_train, feature_names=feat_names)
        else:
            est.fit(X_train, y_train)

        end = time.time()

        # things to save
        rules[d[0]].append(vars(est))

        # loop over metrics
        suffixes = ['_train', '_test']
        datas = [(X_train, y_train), (X_test, y_test)]

        if args.splitting_strategy in {'train-tune-test', 'train-tune-test-lowdata'}:
            suffixes.append('_tune')
            datas.append([X_tune, y_tune])

        metric_results = {}

        for suffix, (X_, y_) in zip(suffixes, datas):
            y_pred = est.predict(X_)
            if args.calc_interactions:
                gt_importance, gt_interaction = get_gt(d[0])
                importance = get_importances(est, X_, y_)
                important_features = get_important_features(importance, len(gt_importance))
                interaction = get_interaction_score(est, X_, y_)
                interacting_features = get_interacting_features(interaction, len(gt_interaction) * 2)
            if args.classification_or_regression == 'classification':
                y_pred_proba = est.predict_proba(X_)
                if y_pred_proba.size == y_.size * 2:  # binary classification with 2 outputs
                    y_pred_proba = y_pred_proba[..., 1]  # take class 1 (for pyGAM, this is skipped)
            for i, (met_name, met) in enumerate(metrics):
                if met is not None:
                    if met_name.startswith("interaction"):
                        if args.calc_interactions:
                            metric_results[met_name + suffix] = met(gt_interaction, interacting_features)
                            metric_results[met_name.replace("interaction", "importance") + suffix] = met(gt_importance,
                                                                                                         important_features)

                    elif args.classification_or_regression == 'regression' \
                            or met_name in ['accuracy', 'f1', 'precision', 'recall']:
                        metric_results[met_name + suffix] = met(y_, y_pred)
                    else:
                        metric_results[met_name + suffix] = met(y_, y_pred_proba)

            # metric_results['interaction' + suffix] = len(interacting_features.difference(gt_interaction)) / len(
            #     interacting_features)
            # metric_results['importance' + suffix] = len(important_features.difference(gt_importance)) / len(
            #     important_features)

        metric_results['complexity'] = util.get_complexity(est)
        metric_results['time'] = end - start
        metric_results.update(model.extra_aggregate_keys)  # add extra keys to aggregate over

        #
        #
        #
        # scores["values"].append((,
        #                      ))
        # scores[d[0]]['interactions'].append(get_interaction_score(est, X_train, y_train))

        for met_name, met_val in metric_results.items():
            colname = met_name
            results[colname].append(met_val)
    return results, rules


def run_comparison(path: str,
                   dataset: Tuple,
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

    results, rules = compare_estimators(estimators=estimators,
                                        dataset=dataset,
                                        metrics=metrics,
                                        args=args)

    estimators_list = [e.name for e in estimators]
    metrics_list = [m[0] for m in metrics]
    df = pd.DataFrame.from_dict(results)
    df['split_seed'] = args.split_seed
    df['estimator'] = estimators_list
    df_rules = pd.DataFrame.from_dict(rules)
    df_rules['split_seed'] = args.split_seed
    df_rules['estimator'] = estimators_list

    # scores_vals = scores["values"]
    # importance = np.stack([s[0] for s in scores_vals], axis=-1)
    # interaction = np.stack([s[1] for s in scores_vals], axis=-1)

    """
    # note: this is only actually a mean when using multiple cv folds
    for met_name, met in metrics:
        colname = f'mean_{met_name}'
        met_df = df.iloc[:, 1:].loc[:, [met_name in col
                                        for col in df.iloc[:, 1:].columns]]
        df[colname] = met_df.mean(axis=1)
    """

    output_dict = {
        # metadata
        'estimators': estimators_list,
        'comparison_datasets': dataset,
        'metrics': metrics_list,
        # "importance":importance,
        # "interaction": interaction,

        # actual values
        'df': df,
        'df_rules': df_rules,
    }
    pkl.dump(output_dict, open(model_comparison_file, 'wb'))


def get_metrics(classification_or_regression: str = 'classification'):
    mutual = [('complexity', None), ('time', None), ("interaction_tpr", interaction_tpr),
              ("interaction_fpr", interaction_fpr), ("interaction_f1", interaction_f1)]
    if classification_or_regression == 'classification':
        return [
                   ('rocauc', roc_auc_score),
                   ('accuracy', accuracy_score),
                   ('f1', f1_score),
                   ('recall', recall_score),
                   ('precision', precision_score),
                   ('avg_precision', average_precision_score),
                   ('best_accuracy', get_best_accuracy),
               ] + mutual
    elif classification_or_regression == 'regression':
        return [
                   ('r2', r2_score),
                   ('explained_variance', explained_variance_score),
                   ('neg_mean_squared_error', mean_squared_error),
               ] + mutual


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
    parser.add_argument('--calc_interactions', action='store_true',
                        help='whether to calculate interactions')
    args = parser.parse_args()

    assert args.splitting_strategy in {
        'train-test', 'train-tune-test', 'train-test-lowdata', 'train-tune-test-lowdata'}

    DATASETS_CLASSIFICATION, DATASETS_REGRESSION, \
    ESTIMATORS_CLASSIFICATION, ESTIMATORS_REGRESSION = config.get_configs(args.config)

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
    elif args.classification_or_regression == 'regression':
        datasets = DATASETS_REGRESSION
        ests = ESTIMATORS_REGRESSION

    metrics = get_metrics(args.classification_or_regression)

    # filter based on args
    if args.dataset:
        datasets = list(filter(lambda x: args.dataset.lower() == x[0].lower(), datasets))  # strict
        # dataset = list(filter(lambda x: args.dataset.lower() in x[0].lower(), dataset)) # flexible
    if args.model:
        #         ests = list(filter(lambda x: args.model.lower() in x[0].name.lower(), ests))
        ests = list(filter(lambda x: args.model.lower() == x[0].name.lower(), ests))

    """
    if args.ensemble:
        ests = get_ensembles_for_dataset(args.dataset, test=args.test)
    else:
        ests = get_estimators_for_dataset(args.dataset, test=args.test)
    """

    if len(ests) == 0:
        raise ValueError('No valid estimators', 'dset', args.dataset, 'models', args.model)
    if len(datasets) == 0:
        raise ValueError('No valid dataset!')
    if args.verbose:
        print('running',
              'dataset', [d[0] for d in datasets],
              'ests', ests)
        print('\tsaving to', args.results_path)
    #         print('\tinput arguments:', args.dataset, [d[0] for d in DATASETS_CLASSIFICATION])

    for dataset in tqdm(datasets):
        path = util.get_results_path_from_args(args, dataset[0])
        for est in ests:
            np.random.seed(1)
            run_comparison(path=path,
                           dataset=dataset,
                           metrics=metrics,
                           estimators=est,
                           args=args)
    print('completed all experiments successfully!')
