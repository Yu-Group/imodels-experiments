import math
import os.path
import pickle as pkl
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
from tqdm import tqdm

from util import remove_x_axis_duplicates, merge_overlapping_curves

dvu.set_style()
mpl.rcParams['figure.dpi'] = 250

cb2 = '#66ccff'
cb = '#1f77b4'
cr = '#cc0000'
cp = '#cc3399'
cy = '#d8b365'
cg = '#5ab4ac'

DIR_FIGS = oj(dirname(os.path.realpath(__file__)), 'figs')


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
            pkl_file = oj('results', 'saps', dset_name, 'train-test/results_aggregated.pkl')
            df = pkl.load(open(pkl_file, 'rb'))['df_mean']
            # print('ks', df.keys())
        else:
            pkl_file = oj('results', 'saps', dset_name, 'train-test/seed0/results_aggregated.pkl')
            df = pkl.load(open(pkl_file, 'rb'))['df']
            suffix = ''
        for _, (name, g) in enumerate(df.groupby('estimator')):
            if name in models_to_include:
                # print('g keys', g.keys())
                x = g['complexity' + suffix].values
                y = g[f'{metric}_test' + suffix].values
                yerr = g[f'{metric}_test' + '_std'].values
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
                plt.errorbar(x[args], y[args], yerr=yerr[args], fmt='.-', **kwargs)
                if show_train:
                    plt.plot(g[f'complexity_train'][args], g[f'{dset_name}_{metric}_train'][args], '.--', **kwargs,
                             label=name + ' (Train)')
                plt.xlabel('Number of rules')
                plt.xlim((0, 20))
                plt.ylabel(
                    dset_name.capitalize().replace('-', ' ') + ' ' + metric.upper().replace('ROC', '').replace('R2',
                                                                                                               '$R^2$'))
        #         if i % C == C - 1:
        if i % C == C - 1:
#             rect = patches.Rectangle((18, 0), 100, 1, linewidth=1, edgecolor='w', facecolor='w', zorder=-4)
#             dvu.line_legend(fontsize=10, xoffset_spacing=0.1, adjust_text_labels=True)
#             ax.add_patch(rect)
            plt.legend()
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
