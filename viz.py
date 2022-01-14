import copy
import math
import os.path
import pickle
import pickle as pkl
import time
from math import ceil
from os.path import dirname
from os.path import join as oj
from typing import List, Dict, Any, Union, Tuple

import dvu
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bartpy import BART, ShrunkBART, ShrunkBARTCV
from imodels import ShrunkTreeRegressorCV, OptimalTreeClassifier
from imodels.tree.gosdt.pygosdt import ShrunkOptimalTreeClassifier
from imodels.util.data_util import get_clean_dataset
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, roc_auc_score, accuracy_score, \
    f1_score, recall_score, precision_score, average_precision_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from tqdm import tqdm

import util
from util import remove_x_axis_duplicates, merge_overlapping_curves
from validate import get_best_accuracy

dvu.set_style()
mpl.rcParams['figure.dpi'] = 250

cb2 = '#66ccff'
cb = '#1f77b4'
cr = '#cc0000'
cp = '#cc3399'
cy = '#d8b365'
cg = '#5ab4ac'

DIR_FIGS = oj(dirname(os.path.realpath(__file__)), 'figs')
is_ssh = "SSH_CONNECTION" in os.environ
R_PATH = "/Users/omerronen/Documents/Phd/tree_shrinkage/imodels-experiments"
if is_ssh:
    R_PATH = "/accounts/campus/omer_ronen/projects/tree_shrink/imodels-experiments"


def pkl_save(f_name, data):
    with open(f'{f_name}.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pkl_load(f_name):
    with open(f'{f_name}.pickle', 'rb') as handle:
        return pickle.load(handle)


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


def plot_bart_comparison(metric='rocauc', datasets=[], seed=None,
                         save_name='fig', show_train=False):
    """Plots curves for different models as a function of complexity

    Params
    ------
    metric: str
        Which metric to plot on y axis
    """
    R, C = ceil(len(datasets) / 3), 3
    plt.figure(figsize=(3 * C, 2.5 * R), facecolor='w')
    met_dict = {"r2": r2_score, "rocauc": roc_auc_score}

    r_rng = [1, 5, 10, 20]

    def _get_preds(mdl, X, is_cls):
        return mdl.predict_proba(X)[..., 1] if is_cls else mdl.predict(X)

    for i, dset in enumerate(tqdm(datasets)):
        performance = {"bart": {r: [] for r in r_rng},
                       "shrunk bart node": {r: [] for r in r_rng},
                       "shrunk bart leaf": {r: [] for r in r_rng},
                       "shrunk bart constant": {r: [] for r in r_rng}}


        for split_seed in tqdm(range(9), colour="green"):
            X, y, feat_names = get_clean_dataset(dset[1], data_source=dset[2])
            is_cls = len(np.unique(y)) == 2

            # implement provided splitting strategy
            splitting_strategy = "train-test"
            # split_seed = 1
            X_train, X_tune, X_test, y_train, y_tune, y_test = (
                util.apply_splitting_strategy(X, y, splitting_strategy, split_seed))
            dset_name = dset[0]
            # bart_org = BARTRegressor(n_chains=6, n_samples=1, n_trees=5)
            # bart_org.fit(X_train, y_train)
            # actual_range = []
            for r in r_rng:
                bart_org = BART(classification=is_cls, n_trees=r, n_samples=1, n_chains=1)
                bart_org.fit(X_train, y_train)
                bart = copy.deepcopy(bart_org)
                # bart = bart.update_complexity(r)
                # bart_c = copy.deepcopy(bart)
                y_test_bart = _get_preds(bart_org, X_test, is_cls)  # bart_org.predict(X_test)
                # tree = RandomForestClassifier()
                # tree.fit(X_train, y_train)
                # y_test_tree = [p[1] for p in tree.predict_proba(X_test)]
                #
                # auc_tree = roc_auc_score(y_test, y_test_tree)
                # auc_bart = roc_auc_score(y_test, y_test_bart)
                # continue

                # print(f"r: {r}, complexity: {bart.sample_complexity}\n")
                # actual_range.append(bart_c.sample_complexity)
                # shrunk_tree = ShrunkBARTRegressor(estimator_=bart, reg_param=0)
                # shrunk_tree.fit(X_train, y_train)
                shrunk_tree = ShrunkBARTCV(estimator_=copy.deepcopy(bart), scheme="node_based")
                shrunk_tree.fit(X_train, y_train)
                shrunk_tree_l = ShrunkBARTCV(estimator_=copy.deepcopy(bart), scheme="leaf_based")
                shrunk_tree_l.fit(X_train, y_train)
                shrunk_tree_c = ShrunkBARTCV(estimator_=copy.deepcopy(bart), scheme="constant")
                shrunk_tree_c.fit(X_train, y_train)

                # shrunk_tree = ShrunkBARTRegressor(estimator_=copy.deepcopy(bart), scheme="node_based", reg_param=1)
                # shrunk_tree.fit(X_train, y_train)
                # shrunk_tree_l = ShrunkBARTRegressor(estimator_=copy.deepcopy(bart), scheme="leaf_based", reg_param=1)
                # shrunk_tree_l.fit(X_train, y_train)
                # shrunk_tree_c = ShrunkBARTRegressor(estimator_=copy.deepcopy(bart), scheme="constant", reg_param=1)
                # shrunk_tree_c.fit(X_train, y_train)
                # m = ShrunkTreeRegressorCV(estimator_=DecisionTreeRegressor(max_leaf_nodes=bart.sample_complexity + 1))
                # m.fit(X_train, y_train)
                y_test_st = _get_preds(shrunk_tree, X_test, is_cls)  # shrunk_tree.predict(X_test) if not is_cls else
                y_test_st_l = _get_preds(shrunk_tree_l, X_test, is_cls)
                y_test_st_c = _get_preds(shrunk_tree_c, X_test, is_cls)
                performance['shrunk bart node'][r].append(met_dict[metric](y_test, y_test_st))
                performance['shrunk bart leaf'][r].append(met_dict[metric](y_test, y_test_st_l))
                performance['shrunk bart constant'][r].append(met_dict[metric](y_test, y_test_st_c))

                performance['bart'][r].append(met_dict[metric](y_test, y_test_bart))
                # performance['shrunk tree'][r].append(met_dict[metric](y_test, m.predict(X_test)))

        def _get_mean_std(method):

            mean_p = [np.mean(performance[method][rng]) for rng in r_rng]
            std_p = [np.std(performance[method][rng]) for rng in r_rng]

            return mean_p, std_p

        bart_perf = _get_mean_std("bart")
        s_bart_perf = _get_mean_std("shrunk bart node")
        s_bart_l_perf = _get_mean_std("shrunk bart leaf")
        s_bart_c_perf = _get_mean_std("shrunk bart constant")
        ax = plt.subplot(R, C, i + 1)

        ax.errorbar(r_rng, bart_perf[0], yerr=bart_perf[1], c="blue", label="BART")
        ax.errorbar(r_rng, s_bart_perf[0], yerr=s_bart_perf[1], c="red", label="Shrunk BART Node")
        ax.errorbar(r_rng, s_bart_l_perf[0], yerr=s_bart_l_perf[1], c="green", label="Shrunk BART Leaf")
        ax.errorbar(r_rng, s_bart_c_perf[0], yerr=s_bart_c_perf[1], c="purple", label="Shrunk BART Constant")
        # ax.errorbar(actual_range, s_tree_perf[0], yerr=s_tree_perf[1], c="green", label=f"Shrunk CART")

        ax.set_xlabel('Number of Trees')
        ax.set_ylabel(
            dset_name.capitalize().replace('-', ' ') + ' ' + metric.upper().replace('ROC', '').replace('R2',
                                                                                                       '$R^2$'))
        if i == 0:
            ax.legend(fontsize=8, loc="upper left")
    savefig(os.path.join(R_PATH, save_name))


def godst_comparison(datasets=[],
                     save_name='godst'):
    """Plots curves for different models as a function of complexity

    Params
    ------
    metric: str
        Which metric to plot on y axis
    """
    # R, C = ceil(len(datasets) / 3), 3
    # plt.figure(figsize=(3 * C, 2.5 * R), facecolor='w')
    #
    met_dict = get_metrics()

    expr_data = {}

    for i, dset in enumerate(tqdm(datasets)):
        perf_ds = {"godst": {"auc": [], "models": []},
                   "godst shrunk": {"auc": [], "models": []}}

        for split_seed in np.arange(0, 9):
            # print(split_seed)
            X, y, feat_names = get_clean_dataset(dset[1], data_source=dset[2])

            # implement provided splitting strategy
            splitting_strategy = "train-test"
            # split_seed = 1
            X_train, X_tune, X_test, y_train, y_tune, y_test = (
                util.apply_splitting_strategy(X, y, splitting_strategy, split_seed))
            dset_name = dset[0]
            # bart_org = BARTRegressor(n_chains=6, n_samples=1, n_trees=5)
            # bart_org.fit(X_train, y_train)
            # actual_range = []
            clf = ExtraTreesClassifier(n_estimators=50)
            clf = clf.fit(X, y)

            model = SelectFromModel(clf, prefit=True, max_features=5)
            X_train = model.transform(X_train)
            X_test = model.transform(X_test)
            # n_features = np.minimum(10, X.shape[-1])
            # X_train = X_train[0:sz, 0:n_features]
            # y_train = y_train[0:sz]
            # X_test = X_test[..., 0:n_features]
            # print(X_train.shape[1])
            # for r in r_rng:
            godst = OptimalTreeClassifier(regularization=0.04)
            s_gosdt = time.time()
            godst.fit(X_train, y_train)
            time_gosdt = time.time() - s_gosdt
            perf_ds["godst"]["models"].append(godst)
            godst_shrunk = ShrunkOptimalTreeClassifier(copy.deepcopy(godst))
            godst_shrunk.fit(X_train, y_train)
            perf_ds["godst shrunk"]["models"].append(godst_shrunk)

            preds_godst = godst.predict_proba_new(X_test)
            preds_shrunk = godst_shrunk.predict_proba(X_test)

            perf_ds['godst']["auc"].append(met_dict[metric](y_test, preds_godst))
            perf_ds['godst shrunk']["auc"].append(met_dict[metric](y_test, preds_shrunk))

        expr_data[dset_name] = perf_ds

        # ax = plt.subplot(R, C, i + 1)
        #
        # # print(performance)
        #
        # bp1 = ax.boxplot(perf_ds['godst']["auc"], positions=[1], widths=0.35,
        #                  patch_artist=True, boxprops=dict(facecolor="C0"))
        # bp2 = ax.boxplot(perf_ds['godst shrunk']["auc"], positions=[2], widths=0.35,
        #                  patch_artist=True, boxprops=dict(facecolor="C2"))
        #
        # ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['godst', 'godst shrunk'], loc='upper right')
        #
        # ax.set_ylabel(
        #     dset_name.capitalize().replace('-', ' ') + ' ' + metric.upper().replace('ROC', '').replace('R2',
        #                                                                                                '$R^2$'))
        # if i == 0:
        #     ax.legend(fontsize=8, loc="upper left")
    # savefig(os.path.join(R_PATH, save_name))
    pkl_save(os.path.join(R_PATH, save_name), expr_data)


# def _get_node_values(node):
#     left = []
#     right = []
#     feature= []
#     threshold =[]
#     def _get_vals_rec(node):
#         if type(node.left_child) == LeafNode and type(node.right_child) == LeafNode:
#             left.append()


# def convert_bart_tree(tree):
#     skl_tree = tree.DecisionTreeRegressor()
#     X = np.random.normal(size=(n, p))
#     y = np.random.normal(size=n)
#     skl_tree.fit(X,y)
#     for d_node in tree.decision_nodes:
#         skl_tree.


def plot_comparisons(metric='rocauc', datasets=[],
                     models_to_include=['SAPS', 'CART'], seed=None,
                     save_name='fig', show_train=False):
    """Plots curves for different models as a function of complexity

    Params
    ------
    metric: str
        Which metric to plot on y axis
    """
    R, C = ceil(len(datasets) / 3), 3
    plt.figure(figsize=(3 * C, 2.5 * R), facecolor='w')

    COLORS = {
        'SAPS': 'black',
        'CART': 'orange',  # cp,
        'Rulefit': 'green',
        'C45': cb,
        'CART_(MSE)': 'orange',
        'CART_(MAE)': cg,
        'SAPS_(Reweighted)': cg,
        'SAPS_(Include_Linear)': cb,
        "RF_(MSE)": "blue"
    }

    for i, dset in enumerate(tqdm(datasets)):
        if isinstance(dset, str):
            dset_name = dset
        elif isinstance(dset, tuple):
            dset_name = dset[0]
        #         try:
        ax = plt.subplot(R, C, i + 1)

        suffix = '_mean'
        if seed is None:
            pkl_file = oj(R_PATH, 'results', 'shrinkage_o', dset_name, 'train-test/results_aggregated.pkl')
            df = pkl.load(open(pkl_file, 'rb'))['df_mean']
            # print('ks', df.keys())
        else:
            pkl_file = oj(R_PATH, 'results', 'saps', dset_name, 'train-test/seed0/results_aggregated.pkl')
            df = pkl.load(open(pkl_file, 'rb'))['df']
            suffix = ''
        for _, (name, g) in enumerate(df.groupby('estimator')):
            if name in models_to_include:
                # print('g keys', g.keys())
                x = g['complexity' + suffix].values
                metric_name = f'{metric}_test' if metric != "time" else metric
                y = g[metric_name + suffix].values
                yerr = g[metric_name + '_std'].values
                args = np.argsort(x)
                #                 print(x[args])
                #                 if i % C == C - 1:
                #                     for cutoff in args:
                #                         if args[cutoff] >= 20:
                #                             break
                #                     args = args[:cutoff - 1]
                alpha = 1.0 if 'SAPS' == name else 0.35
                lw = 2 if 'SAPS' == name else 1.5
                kwargs = dict(color=COLORS.get(name, 'gray'),
                              alpha=alpha,
                              lw=lw,
                              label=name.replace('_', ' ').replace('C45', 'C4.5'),
                              zorder=-5,
                              )
                #                 print(g.keys())
                #                 plt.plot(x[args], y[args], '.-', **kwargs)
                ax.errorbar(x[args], y[args], yerr=yerr[args], fmt='.-', **kwargs)
                if show_train:
                    plt.plot(g[f'complexity_train'][args], g[f'{dset_name}_{metric}_train'][args], '.--', **kwargs,
                             label=name + ' (Train)')
                ax.xlabel('Number of rules')
                # plt.xlim((0, 20))
                ax.ylabel(
                    dset_name.capitalize().replace('-', ' ') + ' ' + metric.upper().replace('ROC', '').replace('R2',
                                                                                                               '$R^2$'))
        #         if i % C == C - 1:
        if i % C == C - 1:
            #             rect = patches.Rectangle((18, 0), 100, 1, linewidth=1, edgecolor='w', facecolor='w', zorder=-4)
            #             dvu.line_legend(fontsize=10, xoffset_spacing=0.1, adjust_text_labels=True)
            #             ax.add_patch(rect)
            ax.legend()
    #         except:
    #             print('skipping', dset_name)
    savefig(save_name)


def savefig(fname):
    os.makedirs(DIR_FIGS, exist_ok=True)
    plt.tight_layout()
    # print(oj(DIR_FIGS, fname + '.pdf'))
    plt.savefig(oj(DIR_FIGS, fname + '.pdf'))


def get_x_and_y(result_data: pd.Series, x_col: str, y_col: str, test=False) -> Tuple[np.array, np.array]:
    if test and result_data.index.unique().shape[0] > 1:
        return merge_overlapping_curves(result_data, y_col)

    complexities = result_data[x_col]
    rocs = result_data[y_col]
    complexity_sort_indices = complexities.argsort()
    x = complexities[complexity_sort_indices]
    y = rocs[complexity_sort_indices]
    return remove_x_axis_duplicates(x.values, y.values)


def viz_comparison_val_average(result: Dict[str, Any], metric: str = 'mean_ROCAUC') -> None:
    '''Plot dataset-averaged y_column vs dataset-averaged complexity for different hyperparameter settings
    of a single model, including zoomed-in plot of overlapping region
    '''
    result_data = result['df']
    result_estimators = result['estimators']
    x_column = 'mean_complexity'
    y_column = metric
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    for est in np.unique(result_estimators):

        est_result_data = result_data[result_data.index.str.fullmatch(est)]
        x, y = get_x_and_y(est_result_data, x_column, y_column)
        axes[0].plot(x, y, marker='o', markersize=4, label=est.replace('_', ' '))

        meta_auc_df = result['meta_auc_df']
        area = meta_auc_df.loc[est, y_column + '_auc']
        label = est.split(' - ')[1]
        if area != 0:
            label += f' {y_column} AUC: {area:.3f}'
        axes[1].plot(x, y, marker='o', markersize=4, label=label.replace('_', ' '))

    axes[0].set_title(f'{metric} across comparison datasets')
    axes[1].set_xlim(meta_auc_df.iloc[0][f'{x_column}_lb'], meta_auc_df.iloc[0][f'{x_column}_ub'])
    axes[0].set_xlim(0, 40)
    axes[1].set_title('Overlapping, low (<30) complexity region only')

    for ax in axes:
        ax.set_xlabel('complexity score')
        ax.set_ylabel(y_column)
        ax.legend(frameon=False, handlelength=1)
        # dvu.line_legend(fontsize=10, ax=ax)
    plt.tight_layout()


def viz_comparison_test_average(results: List[Dict[str, Any]],
                                metric: str = 'mean_rocauc',
                                line_legend: bool = False) -> None:
    '''Plot dataset-averaged y_column vs dataset-averaged complexity for different models
    '''
    x_column = 'mean_complexity'
    y_column = metric
    for result in results:
        result_data = result['df']
        est = result['estimators'][0].split(' - ')[0]
        x, y = get_x_and_y(result_data, x_column, y_column, test=True)
        linestyle = '--' if 'stbl' in est else '-'
        plt.plot(x, y, marker='o', linestyle=linestyle, markersize=2, linewidth=1, label=est.replace('_', ' '))
    plt.xlim(0, 40)
    plt.xlabel('complexity score', size=8)
    plt.ylabel(y_column, size=8)
    plt.title(f'{metric} across comparison datasets', size=8)
    if line_legend:
        dvu.line_legend(fontsize=8, adjust_text_labels=True)
    else:
        plt.legend(frameon=False, handlelength=1, fontsize=8)


def viz_comparison_datasets(result: Union[Dict[str, Any], List[Dict[str, Any]]],
                            y_column: str = 'ROCAUC',
                            cols=3,
                            figsize=(14, 10),
                            line_legend: bool = False,
                            datasets=None,
                            test=False) -> None:
    '''Plot y_column vs complexity for different datasets and models (not averaged)
    '''
    if test:
        results_data = pd.concat([r['df'] for r in result])
        results_estimators = [r['estimators'][0] for r in result]
        results_datasets = result[0]['comparison_datasets']
    else:
        results_data = result['df']
        results_estimators = np.unique(result['estimators'])
        results_datasets = result['comparison_datasets']

    if datasets is None:
        datasets = list(map(lambda x: x[0], results_datasets))
    n_rows = int(math.ceil(len(datasets) / cols))
    plt.figure(figsize=figsize)
    for i, dataset in enumerate(datasets):
        plt.subplot(n_rows, cols, i + 1)

        for est in np.unique(results_estimators):
            est_result_data = results_data[results_data.index.str.fullmatch(est)]
            x, y = get_x_and_y(est_result_data, dataset + '_complexity', dataset + f'_{y_column}')

            linestyle = '--' if 'stbl' in est else '-'
            plt.plot(x, y, marker='o', linestyle=linestyle, markersize=4, label=est.replace('_', ' '))

        plt.xlim(0, 40)
        plt.xlabel('complexity score')
        plt.ylabel(y_column)
        if line_legend:
            dvu.line_legend(fontsize=14,
                            adjust_text_labels=False,
                            xoffset_spacing=0,
                            extra_spacing=0)
        else:
            plt.legend(frameon=False, handlelength=1)
        plt.title(f'dataset {dataset}')
    #     plt.subplot(n_rows, cols, 1)
    plt.tight_layout()


if __name__ == '__main__':
    DATASETS_REGRESSION = [
        # leo-breiman paper random forest uses some UCI datasets as well
        # pg 23: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
        ('friedman1', 'friedman1', 'synthetic'),
        ('friedman2', 'friedman2', 'synthetic'),
        ('friedman3', 'friedman3', 'synthetic'),
        ("diabetes-regr", "diabetes", 'sklearn'),
        ("geographical-music", "4544_GeographicalOriginalofMusic", "pmlb"),

        ("red-wine", "wine_quality_red", "pmlb"),
        ('abalone', '183', 'openml'),
        ("satellite-image", "294_satellite_image", 'pmlb'),

        ("california-housing", "california_housing", 'sklearn'),  # this replaced boston-housing due to ethical issues
        # ("echo-months", "1199_BNG_echoMonths", 'pmlb')
        # ("breast-tumor", "1201_BNG_breastTumor", 'pmlb'),  # this one is v big (100k examples)

    ]
    DATASETS_CLASSIFICATION = [
        # classification datasets from original random forests paper
        # page 9: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
        # ("sonar", "sonar", "pmlb"),
        ("heart", "heart", 'imodels'),
        ("breast-cancer", "breast_cancer", 'imodels'),
        ("haberman", "haberman", 'imodels'),
        ("ionosphere", "ionosphere", 'pmlb'),
        ("diabetes", "diabetes", "pmlb"),
        # # ("liver", "8", "openml"), # note: we omit this dataset bc it's label was found to be incorrect (see caveat here: https://archive.ics.uci.edu/ml/datasets/liver+disorders#:~:text=The%207th%20field%20(selector)%20has%20been%20widely%20misinterpreted%20in%20the%20past%20as%20a%20dependent%20variable%20representing%20presence%20or%20absence%20of%20a%20liver%20disorder.)
        # # ("credit-g", "credit_g", 'imodels'), # like german-credit, but more feats
        # ("german-credit", "german", "pmlb"),
        #
        # # clinical-decision rules
        # # ("iai-pecarn", "iai_pecarn.csv", "imodels"),
        #
        # # popular classification datasets used in rule-based modeling / fairness
        # # page 7: http://proceedings.mlr.press/v97/wang19a/wang19a.pdf
        ("juvenile", "juvenile_clean", 'imodels'),
        ("recidivism", "compas_two_year_clean", 'imodels'),
        ("credit", "credit_card_clean", 'imodels'),
        # ("readmission", 'readmission_clean', 'imodels'),  # v big
    ]
    plot_bart_comparison("r2", datasets=DATASETS_REGRESSION, save_name="bart_reg")
    plot_bart_comparison("rocauc", datasets=DATASETS_CLASSIFICATION, save_name="bart_cls")

    # godst_comparison(datasets=DATASETS_CLASSIFICATION)
