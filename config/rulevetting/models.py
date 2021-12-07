import functools
import itertools
from collections import OrderedDict
from typing import List

import numpy as np
from imodels import (
    SkopeRulesClassifier as skope, RuleFitClassifier as rfit, BoostedRulesClassifier as brs,
    SaplingSumClassifier as saps, GreedyRuleListClassifier as grl
)
from sklearn.ensemble import RandomForestClassifier as rf, GradientBoostingClassifier as gb
from sklearn.tree import DecisionTreeClassifier as dt

from util import Model2


def grid_to_kwargs(grid: OrderedDict) -> List[dict]:
    all_kwargs = []

    for args_combo in itertools.product(*grid.values()):
        curr_kwargs = {list(grid.keys())[i]: args_combo[i] for i in range(len(grid))}
        all_kwargs.append(curr_kwargs)

    return all_kwargs


cart_grid = OrderedDict({
    'max_leaf_nodes': np.arange(1, 30),
    'class_weight': [
        {0: 1, 1: 10},
        {0: 1, 1: 100},
        {0: 1, 1: 100}
    ],
    'criterion': ['gini', 'entropy'],
})


random_forest_grid = OrderedDict({
    'n_estimators': np.arange(1, 10),
    'max_samples': [0.8, 0.9, 1.0],
    'max_depth': [2, 3]
})

gradient_boosting_grid = OrderedDict({
    'n_estimators': np.arange(1, 10),
    'loss': ['deviance', 'exponential'],
    'max_depth': [2, 3]
})

skope_rules_grid = OrderedDict({
    'n_estimators': np.arange(1, 20),
    'precision_min': [0.2, 0.3, 0.4],
    'recall_min': [0.2, 0.3, 0.4],
    'max_depth': [2, 3]
})

rulefit_grid = OrderedDict({
    'n_estimators': np.arange(1, 20),
    'alpha': [1.0, 2.0, 5.0, 13.0, 20.0, 50.0, 100.0],
    'random_state': [0],
    'max_rules': [None],
    'include_linear': [False]
})

brs_grid = OrderedDict({
    'n_estimators': np.arange(1, 13),
    'estimator': [
        functools.partial(dt, min_samples_split=0.0001, max_depth=2),
        functools.partial(dt, min_samples_split=0.001, max_depth=2),
        functools.partial(dt, min_samples_split=0.01, max_depth=2),
        functools.partial(dt, min_samples_split=0.05, max_depth=2),
        functools.partial(dt, max_depth=1),
    ]
})

saps_grid = OrderedDict({
    'max_rules': np.arange(1, 30),
})

grl_grid = OrderedDict({
    'max_depth': np.arange(1, 15),
    'class_weight': [
        {0: 1, 1: 10},
        {0: 1, 1: 100},
        {0: 1, 1: 1000}
    ],
    'criterion': ['neg_corr']
})


ESTIMATORS_CLASSIFICATION = [

    [Model2('cart', dt, kw) for kw in grid_to_kwargs(cart_grid)],
    [Model2('random_forest', rf, kw) for kw in grid_to_kwargs(random_forest_grid)],
    [Model2('gradient_boosting', gb, kw) for kw in grid_to_kwargs(gradient_boosting_grid)],
    [Model2('skope_rules', skope, kw) for kw in grid_to_kwargs(skope_rules_grid)],
    [Model2('rulefit', rfit, kw) for kw in grid_to_kwargs(rulefit_grid)],
    [Model2('brs', brs, kw) for kw in grid_to_kwargs(brs_grid)],
    [Model2('saps', saps, kw) for kw in grid_to_kwargs(saps_grid)],
    [Model2('grl', grl, kw) for kw in grid_to_kwargs(grl_grid)]

]
ESTIMATORS_REGRESSION = []
