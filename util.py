import os
import warnings
from functools import partial
from os.path import dirname
from os.path import join as oj
from typing import Any, Dict, Tuple

import numpy as np
from sklearn import model_selection
from sklearn.base import BaseEstimator

from imodels.util.tree import compute_tree_complexity

DATASET_PATH = oj(dirname(os.path.realpath(__file__)), 'data')


class Model:
    def __init__(self,
                 name: str, cls,
                 vary_param: str = None, vary_param_val: Any = None,
                 fixed_param: str = None, fixed_param_val: Any = None,
                 other_params: Dict[str, Any] = {}):
        self.name = name
        self.cls = cls
        self.fixed_param = fixed_param
        self.fixed_param_val = fixed_param_val
        self.vary_param = vary_param
        self.vary_param_val = vary_param_val
        self.kwargs = {}
        if self.vary_param is not None:
            self.kwargs[self.vary_param] = self.vary_param_val
        if self.fixed_param is not None:
            self.kwargs[self.fixed_param] = self.fixed_param_val
        self.kwargs = {**self.kwargs, **other_params}

    def __repr__(self):
        return self.name


def get_results_path_from_args(args, dataset):
    """Gets path of directory in which model result pkls will be stored.
    Path structure:
        results/(saps|stablerules|shrinkage)/{dataset}/{splitting strategy}
    """
    path = args.results_path
    path = oj(path, args.config)
    path = oj(path, dataset)
    path = oj(path, args.splitting_strategy)
    path = oj(path, 'seed' + str(args.split_seed))
    os.makedirs(path, exist_ok=True)
    return path


def get_best_model_under_complexity(c: int, model_name: str,
                                    model_cls: BaseEstimator,
                                    dataset: str,
                                    curve_params: list = None,
                                    metric: str = 'mean_rocauc',
                                    kwargs: dict = {},
                                    prefix: str = 'val',
                                    easy: bool = False) -> BaseEstimator:
    # init_models = []
    # for m_name, m_cls in models:
    result = get_comparison_result(MODEL_COMPARISON_PATH, model_name, dataset=dataset, prefix=prefix)
    df, auc_metric = result['df'], result['meta_auc_df'][f'{metric}_auc']

    if curve_params:
        # specify which curve to use
        if type(df.iloc[:, 1][0]) is partial:
            df_best_curve = df[df.iloc[:, 1].apply(lambda x: x.keywords['min_samples_split']).isin(curve_params)]
        else:
            df_best_curve = df[df.iloc[:, 1].isin(curve_params)]

    else:
        # detect which curve to use
        df_best_curve = df[df.index == auc_metric.idxmax()]

    df_under_c = df_best_curve[df_best_curve['mean_complexity'] < c]
    if df_under_c.shape[0] == 0:
        warnings.warn(f'{model_name} skipped for complexity limit {c}')
        return None

    best_param = df_under_c.iloc[:, 0][df_under_c[metric].argmax()]
    kwargs[df_under_c.columns[0]] = best_param

    # if there is a second param that was varied
    if auc_metric.shape[0] > 1:
        kwargs[df_under_c.columns[1]] = df_under_c.iloc[0, 1]

    return model_cls(**kwargs)


def remove_x_axis_duplicates(x: np.array, y: np.array) -> Tuple[np.array, np.array]:
    unique_arr, inds, counts = np.unique(x, return_index=True, return_counts=True)

    y_for_unique_x = []
    for i, ind in enumerate(inds):
        y_for_unique_x.append(y[ind:ind + counts[i]].max())

    return unique_arr, np.array(y_for_unique_x)


def merge_overlapping_curves(test_mul_curves, y_col):
    final_x = []
    final_y = []
    curves = test_mul_curves.index.unique()

    start_compl = 0
    for i in range(curves.shape[0]):
        curr_x = test_mul_curves[test_mul_curves.index == curves[i]]['mean_complexity']
        curr_y = test_mul_curves[test_mul_curves.index == curves[i]][y_col]
        curr_x, curr_y = curr_x[curr_x.argsort()], curr_y[curr_x.argsort()]
        curr_x, curr_y = remove_x_axis_duplicates(curr_x, curr_y)
        curr_x, curr_y = curr_x[curr_x >= start_compl], curr_y[curr_x >= start_compl]

        if i != curves.shape[0] - 1:
            next_x = test_mul_curves[test_mul_curves.index == curves[i + 1]]['mean_complexity']
            next_y = test_mul_curves[test_mul_curves.index == curves[i + 1]][y_col]
            next_x, next_y = next_x[next_x.argsort()], next_y[next_x.argsort()]
            next_x, next_y = remove_x_axis_duplicates(next_x, next_y)

        found_switch_point = False
        for j in range(curr_x.shape[0] - 1):

            final_x.append(curr_x[j])
            final_y.append(curr_y[j])

            if i != curves.shape[0] - 1:

                next_x_next_val = next_x[next_x > curr_x[j]][0]
                next_y_next_val = next_y[next_x > curr_x[j]][0]
                curr_x_next_val = curr_x[j + 1]
                curr_y_next_val = curr_y[j + 1]

                if next_y_next_val > curr_y_next_val and next_x_next_val - curr_x_next_val <= 5:
                    start_compl = next_x_next_val
                    found_switch_point = True
                    break

        if not found_switch_point:
            return np.array(final_x), np.array(final_y)

    return np.array(final_x), np.array(final_y)


def get_complexity(estimator: BaseEstimator) -> float:
    """Get complexity for any given estimator
    """
    if hasattr(estimator, 'complexity_'):
        return estimator.complexity_
    else:
        # sklearn ensembles have estimator.estimators_
        # RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
        estimators = None
        if hasattr(estimator, 'estimators_'):
            estimators = estimator.estimators_
        elif hasattr(estimator, 'estimator_'):  # ShrunkTreeCV
            if hasattr(estimator.estimator_, 'estimators_'):  # ensemble passed
                estimators = estimator.estimator_.estimators_
            elif hasattr(estimator.estimator_, 'tree_'):  # tree passed
                estimators = [estimator.estimator_]

        if estimators is None:
            raise Warning('Dont know how to compute complexity for ' + str(estimator))

        complexity = 0
        for tree in estimators:
            if isinstance(tree, np.ndarray):
                tree = tree[0]
            complexity += compute_tree_complexity(tree.tree_)
        return complexity


def apply_splitting_strategy(X: np.ndarray,
                             y: np.ndarray,
                             splitting_strategy: str,
                             split_seed: str) -> Tuple[Any, Any, Any, Any, Any, Any]:
    if splitting_strategy in {'train-test-lowdata', 'train-tune-test-lowdata'}:
        test_size = 0.90  # X.shape[0] - X.shape[0] * 0.1
    else:
        test_size = 0.2

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size, random_state=split_seed)
    X_tune = None
    y_tune = None

    if splitting_strategy in {'train-tune-test', 'train-tune-test-lowdata'}:
        X_train, X_tune, y_train, y_tune = model_selection.train_test_split(
            X_train, y_train, test_size=0.2, random_state=split_seed)

    return X_train, X_tune, X_test, y_train, y_tune, y_test
