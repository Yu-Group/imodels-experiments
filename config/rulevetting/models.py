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
from local_models.stable import StableLinearClassifier as stbl


def grid_to_kwargs(grid: OrderedDict) -> List[dict]:
    all_kwargs = []
    all_names = []

    # create a valid kwargs dict for each param combo implied by the grid
    grid_arg_names = list(grid.keys())
    for args_combo in itertools.product(*grid.values()):
        curr_kwargs = {grid_arg_names[i]: args_combo[i] for i in range(len(grid))}
        all_kwargs.append(curr_kwargs)

        # generate a unique run name for each param combo
        name = []
        for i in range(len(grid))[1:]:
            arg_name_repr = grid_arg_names[i]
            arg_repr = args_combo[i]

            # if param wasn't varied it isn't included in the name
            if len(list(grid.values())[i]) == 1:
                continue

            # special handling for class weight dicts
            if arg_name_repr == 'class_weight':
                arg_repr = args_combo[i][1]
            
            # special handling for long stbl param names
            if 'max_complexity' in arg_name_repr:
                arg_name_repr = arg_name_repr.replace('max_complexity', 'mc')

            # special handling for boostedruleset bc it takes partial objects
            elif type(arg_repr) == functools.partial:
                arg_name_repr = 'max_leaf_nodes'
                arg_repr = arg_repr.keywords[arg_name_repr]

            # trim all string param names for brevity
            if type(arg_repr) == str:
                arg_repr = arg_repr[:3]

            name.append(f'{arg_name_repr}_{arg_repr}')
        name = '_'.join(name)
        all_names.append(name)

    return zip(all_kwargs, all_names)


cart_grid = OrderedDict({
    'max_leaf_nodes': np.arange(2, 30),
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
    'recall_min': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [2, 3]
})

rulefit_grid = OrderedDict({
    'max_rules': np.arange(1, 30),
    'n_estimators': [5, 10, 25, 50, 100, 500, 1000],
    'cv': [False],
    'random_state': [0],
    'include_linear': [True]
})

brs_grid = OrderedDict({
    'n_estimators': np.arange(1, 13),
    'estimator': [
        functools.partial(dt, max_leaf_nodes=2, class_weight={0: 1, 1: 6}),
        functools.partial(dt, max_leaf_nodes=3, class_weight={0: 1, 1: 6}),
        functools.partial(dt, max_leaf_nodes=4, class_weight={0: 1, 1: 6}),
    ]
})

saps_grid = OrderedDict({
    'max_rules': np.arange(1, 30),
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


stbl_l2_grid = OrderedDict({
    'max_rules': np.arange(1, 30),
    'max_complexity_brs': [10],
    'max_complexity_rulefit': [5, 15, 30],
    'max_complexity_skope_rules': [5, 15, 30],
    'min_mult': [2, 3],
    'penalty': ['l2'],
    'metric': ['best_spec_0.95_sens'],
    'cv': [False],
    'random_state': [0],
    'submodels': ['rulefit', 'skope_rules', 'brs']
})


stbl_l1_grid = OrderedDict({
    'max_rules': np.arange(1, 30),
    'max_complexity_brs': [10],
    'max_complexity_rulefit': [5, 15, 30],
    'max_complexity_skope_rules': [5, 15, 30],
    'min_mult': [2, 3],
    'penalty': ['l1'],
    'metric': ['best_spec_0.95_sens'],
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
    [Model2('saps', saps, kw, kwid) for (kw, kwid) in grid_to_kwargs(saps_grid)],
    [Model2('grl', grl, kw, kwid) for (kw, kwid) in grid_to_kwargs(grl_grid)],
    [Model2('stbl_l1', stbl, kw, kwid) for (kw, kwid) in grid_to_kwargs(stbl_l1_grid)],
    [Model2('stbl_l2', stbl, kw, kwid) for (kw, kwid) in grid_to_kwargs(stbl_l2_grid)],

]
ESTIMATORS_REGRESSION = []
