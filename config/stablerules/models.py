import functools
import itertools
from collections import OrderedDict
from typing import List

import numpy as np
from imodels import (
    SkopeRulesClassifier as skope, RuleFitClassifier as rfit, BoostedRulesClassifier as brs,
    GreedyRuleListClassifier as grl, OptimalRuleListClassifier as corels
)
from sklearn.ensemble import RandomForestClassifier as rf, GradientBoostingClassifier as gb
from sklearn.tree import DecisionTreeClassifier as dt

from config.rulevetting.models import grid_to_kwargs
from local_models.stable import StableLinearClassifier as stbl
from util import Model2


cart_grid = OrderedDict({
    'max_leaf_nodes': np.arange(2, 35),
    'class_weight': [
        {0: 1, 1: 1},
        {0: 1, 1: 2},
        {0: 1, 1: 6},
        {0: 1, 1: 10}
    ],
    'criterion': ['gini', 'entropy'],
})

random_forest_grid = OrderedDict({
    'n_estimators': np.arange(1, 15),
    'max_samples': [0.8, 0.9, 1.0],
    'max_depth': [2, 3]
})

gradient_boosting_grid = OrderedDict({
    'n_estimators': np.arange(1, 15),
    'loss': ['deviance', 'exponential'],
    'max_depth': [2, 3]
})

skope_rules_grid = OrderedDict({
    'n_estimators': np.arange(1, 50),
    'precision_min': [0.01, 0.1],
    'recall_min': [0.01, 0.1, 0.2],
    'max_depth': [2, 3]
})

rulefit_grid = OrderedDict({
    'max_rules': np.arange(1, 30),
    'n_estimators': [5, 10, 25, 50, 100, 500],
    'cv': [False],
    'random_state': [0],
    'include_linear': [True]
})

brs_grid = OrderedDict({
    'n_estimators': np.arange(1, 15),
    'estimator': [
        functools.partial(dt, max_leaf_nodes=2, class_weight={0: 1, 1: 1}),
        functools.partial(dt, max_leaf_nodes=3, class_weight={0: 1, 1: 1}),
        functools.partial(dt, max_leaf_nodes=4, class_weight={0: 1, 1: 1}),
    ]
})

grl_grid = OrderedDict({
    'max_depth': np.arange(1, 15),
    'class_weight': [
        {0: 1, 1: 1},
        {0: 1, 1: 2},
        {0: 1, 1: 6},
        {0: 1, 1: 10}
    ],
    'criterion': ['neg_corr']
})

corels_grid = OrderedDict({
    'c': np.logspace(-4, 4, num=15),
    'policy': ['lower_bound', 'curious'],
    'ablation': [0, 1, 2],
    'n_iter': [10000, 50000, 100000],
})

stbl_l1_grid = OrderedDict({
    'max_rules': np.arange(1, 30),
    'max_complexity': [5, 10, 20, 30, 40, 50],
    'min_mult': [2, 3],
    'penalty': ['l1'],
    'metric': ['best_spec_0.96_sens'],
    'cv': [False],
    'random_state': [0],
    'submodels': [['rulefit', 'skope_rules', 'brs']]
})

stbl_l2_grid = OrderedDict({
    'max_rules': np.arange(1, 30),
    'max_complexity': [5, 10, 20, 30, 40, 50],
    'min_mult': [2, 3],
    'penalty': ['l2'],
    'metric': ['best_spec_0.96_sens'],
    'cv': [False],
    'random_state': [0],
    'submodels': [['rulefit', 'skope_rules', 'brs']]
})


ESTIMATORS_CLASSIFICATION = [

    [Model2('cart', dt, kw, kwid) for (kw, kwid) in grid_to_kwargs(cart_grid)],
    [Model2('random_forest', rf, kw, kwid) for (kw, kwid) in grid_to_kwargs(random_forest_grid)],
    [Model2('gradient_boosting', gb, kw, kwid) for (kw, kwid) in grid_to_kwargs(gradient_boosting_grid)],
    [Model2('skope_rules', skope, kw, kwid) for (kw, kwid) in grid_to_kwargs(skope_rules_grid)],
    [Model2('rulefit', rfit, kw, kwid) for (kw, kwid) in grid_to_kwargs(rulefit_grid)],
    [Model2('brs', brs, kw, kwid) for (kw, kwid) in grid_to_kwargs(brs_grid)],
    # [Model2('saps', saps, kw, kwid) for (kw, kwid) in grid_to_kwargs(saps_grid)],
    [Model2('grl', grl, kw, kwid) for (kw, kwid) in grid_to_kwargs(grl_grid)],
    [Model2('corels', corels, kw, kwid) for (kw, kwid) in grid_to_kwargs(corels_grid)],
    [Model2('stbl_l1', stbl, kw, kwid) for (kw, kwid) in grid_to_kwargs(stbl_l1_grid)],
    [Model2('stbl_l2', stbl, kw, kwid) for (kw, kwid) in grid_to_kwargs(stbl_l2_grid)],

]

ESTIMATORS_REGRESSION = []
