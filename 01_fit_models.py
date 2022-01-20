import argparse
import copy
import os
import pickle as pkl
import time
import warnings
from collections import defaultdict
from os.path import join as oj
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from imodels import OptimalTreeClassifier
from imodels.tree.gosdt.pygosdt import ShrunkOptimalTreeClassifier, ShrunkOptimalTreeClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, \
    RandomForestRegressor, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, recall_score, \
    precision_score, r2_score, explained_variance_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel
from tqdm import tqdm

import config
import util
from imodels.util.data_util import get_clean_dataset
from util import ModelConfig
from validate import get_best_accuracy

warnings.filterwarnings("ignore", message="Bins whose width")


def compare_estimators(estimators: List[ModelConfig],
                       datasets: List[Tuple],
                       metrics: List[Tuple[str, Callable]],
                       args, ) -> Tuple[dict, dict]:
    """Calculates results given estimators, datasets, and metrics.
    Called in run_comparison
    """
    if type(estimators) != list:
        raise Exception("First argument needs to be a list of Models")
    if type(metrics) != list:
        raise Exception("Argument metrics needs to be a list containing ('name', callable) pairs")

    # initialize results with metadata
    results = defaultdict(lambda: [])
    # estimators = [ModelConfig("OptimalTreeClassifier", OptimalTreeClassifier, "regularization", 0.05),
    #               ModelConfig("ShrunkOptimalTreeClassifier", cls=ShrunkOptimalTreeClassifier)]
    # for e in estimators:
    #     kwargs: dict = e.kwargs  # dict
    #     for k in kwargs:
    #         results[k].append(kwargs[k])
    rules = results.copy()

    # loop over datasets
    for d in datasets:
        if args.verbose:
            print("\tdataset", d[0], 'ests', estimators)
        X, y, feat_names = get_clean_dataset(d[1], data_source=d[2])

        # implement provided splitting strategy
        X_train, X_tune, X_test, y_train, y_tune, y_test = (
            util.apply_splitting_strategy(X, y, args.splitting_strategy, args.split_seed))
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True, max_features=5)
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)

        # loop over estimators
        for model in tqdm(estimators, leave=False):
            # print('kwargs', model.kwargs)
            if model.name == "OptimalTreeClassifier":
                est_gosdt = model.cls(**model.kwargs)
                est = est_gosdt
            else:
                est = model.cls(copy.deepcopy(est_gosdt))
            # print(est.criterion)
            #
            # sklearn_baselines = {
            #     RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier,
            #     RandomForestRegressor, GradientBoostingRegressor, DecisionTreeRegressor}

            start = time.time()
            # ss = 70
            # indx = np.random.choice(np.arange(0, len(y_train)), ss)
            # est.fit(X_train[indx, ...], y_train[indx])
            est.fit(X_train, y_train)
            #
            # if type(est) in sklearn_baselines or True:
            #     est.fit(X_train, y_train)
            # else:
            #     est.fit(X_train, y_train, feature_names=feat_names)
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
                # print('best param', est.reg_param)
                if args.classification_or_regression == 'classification':
                    y_pred_proba = est.predict_proba(X_)[..., 1]
                    # print(f"preds {suffix}: {np.unique(y_pred_proba)}")

                for i, (met_name, met) in enumerate(metrics):
                    if met is not None:
                        if args.classification_or_regression == 'regression' \
                                or met_name in ['accuracy', 'f1', 'precision', 'recall']:
                            metric_results[met_name + suffix] = met(y_, y_pred)
                        else:
                            # print(y_pred_proba)
                            metric_results[met_name + suffix] = met(y_, y_pred_proba)
                            # print(f"{model.name} {suffix}: {met_name}: {metric_results[met_name + suffix]},"
                            #       f" preds: {np.mean(y_pred_proba)}")

            metric_results['complexity'] = util.get_complexity(est)
            metric_results['time'] = end - start
            print(f"{model.name}: {metric_results}")

            for met_name, met_val in metric_results.items():
                colname = met_name
                results[colname].append(met_val)
    return results, rules


def run_comparison(path: str,
                   datasets: List[Tuple],
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
                                        datasets=datasets,
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
        'comparison_datasets': datasets,
        'metrics': metrics_list,

        # actual values
        'df': df,
        'df_rules': df_rules,
    }
    # print(output_dict)
    pkl.dump(output_dict, open(model_comparison_file, 'wb'))


def get_metrics(classification_or_regression: str = 'classification'):
    mutual = [('complexity', None), ('time', None)]
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
    parser.add_argument('--reg', type=float,
                        default=0.05)
    args = parser.parse_args()

    assert args.splitting_strategy in {
        'train-test', 'train-tune-test', 'train-test-lowdata', 'train-tune-test-lowdata'}

    DATASETS_CLASSIFICATION, DATASETS_REGRESSION, \
    _, ESTIMATORS_REGRESSION = config.get_configs(args.config)

    reg = args.reg

    ESTIMATORS_CLASSIFICATION = [ModelConfig("OptimalTreeClassifier", OptimalTreeClassifier, "regularization", reg),
                                 ModelConfig("ShrunkOptimalTreeClassifierCV", ShrunkOptimalTreeClassifierCV)]

    print('dset', args.dataset, [d[0] for d in DATASETS_CLASSIFICATION])
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
        # datasets = list(filter(lambda x: args.dataset.lower() in x[0].lower(), datasets)) # flexible
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
        raise ValueError('No valid datasets!')
    if args.verbose:
        print('running',
              'datasets', [d[0] for d in datasets],
              'ests', ests)
        print('saving to', args.results_path)

    for dataset in tqdm(datasets):
        path = util.get_results_path_from_args(args, dataset[0])
        # for s in [1, 2, 3]:
        np.random.seed(1)
        # p = os.path.join(path, f"seed_{s}")
        # if not os.path.exists(p):
        #     os.mkdir(p)
        run_comparison(path=path,
                       datasets=[dataset],
                       metrics=metrics,
                       estimators=ests,
                       args=args)
    print('completed all experiments successfully!')
