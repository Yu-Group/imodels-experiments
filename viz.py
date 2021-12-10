import math
from typing import List, Dict, Any, Union, Tuple

import dvu
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from os.path import dirname
from util import remove_x_axis_duplicates, merge_overlapping_curves

import os.path
from os.path import join as oj
dvu.set_style()
mpl.rcParams['figure.dpi'] = 250

cb2 = '#66ccff'
cb = '#1f77b4'
cr = '#cc0000'
cp = '#cc3399'
cy = '#d8b365'
cg = '#5ab4ac'

DIR_FIGS = oj(dirname(os.path.realpath(__file__)), 'figs')


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


def viz_model_curves_validation(
    ax: plt.Axes,
    result: dict[str, Any],
    suffix: str,
    metric: str = 'rocauc',
    curve_id: str = None) -> None:

    df = result['df']
    if curve_id:
        curve_ids = [curve_id]
    else:
        curve_ids = df['curve_id'].unique()
    dataset = result['dataset']
    
    if suffix == 'test':
        x_column = 'complexity_train'
    else:
        x_column = 'complexity_' + suffix
    y_column = f'{metric}_' + suffix

    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    min_complexity_all_curves = float('inf')
    for curve_id in curve_ids:
        curr_curve_df = df[df['curve_id'] == curve_id]
        curr_est_name = curr_curve_df.index[0]

        x, y = get_x_and_y(curr_curve_df, x_column, y_column)
        min_complexity_all_curves = min(min_complexity_all_curves, x[0])

        # meta_auc_df = result['meta_auc_df']
        # area = meta_auc_df.loc[est, y_column + '_auc']
        # label = est.split(' - ')[1]
        # if area != 0:
        #     label += f' {y_column} AUC: {area:.3f}'
        # axes[0].plot(x, y, marker='o', markersize=4, label=curve_id)
        # axes[1].plot(x, y, marker='o', markersize=4, label=curve_id)

        if len(curve_ids) == 1:
            label = curr_est_name
        else:
            label = curve_id

        ax.plot(x, y, marker='o', markersize=4, label=label)

    # axes[0].set_title(f'{metric} vs. complexity, {curr_est_name} on {dataset}')
    # axes[1].set_xlim(meta_auc_df.iloc[0][f'{x_column}_lb'], meta_auc_df.iloc[0][f'{x_column}_ub'])
    # axes[1].set_xlim(0, 20)
    # axes[0].set_xlim(0, 40)
    # axes[1].set_title('Low (<20) complexity region only')

    ax.set_xlim(0, 30)
    if suffix != 'test':
        est_name_title = curr_est_name
    else:
        est_name_title = 'all'
    ax.set_title(f'{metric} vs. complexity, {est_name_title} on {dataset}')
    ax.set_xlabel('complexity score')
    ax.set_ylabel(y_column)
    ax.legend(frameon=False, handlelength=1)

    # for ax in axes:
    #     ax.set_xlabel('complexity score')
    #     ax.set_ylabel(y_column)
    #     ax.legend(frameon=False, handlelength=1)
    #     dvu.line_legend(fontsize=10, ax=ax)
    
    # axes[0].set_xlabel('complexity score')
    # axes[0].set_ylabel(y_column)
    # axes[0].legend(frameon=False, handlelength=1)
    # dvu.line_legend(fontsize=10, ax=axes[0])

    # plt.tight_layout()


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
                            suffix: str = 'train',
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
        datasets = [d.name for d in results_datasets]
    n_rows = int(math.ceil(len(datasets) / cols))
    plt.figure(figsize=figsize)
    for i, dataset in enumerate(datasets):
        plt.subplot(n_rows, cols, i + 1)

        for est in np.unique(results_estimators):
            est_result_data = results_data[results_data.index.str.fullmatch(est)]
            x, y = get_x_and_y(est_result_data, dataset + '_complexity_' + suffix, dataset + f'_{y_column}_' + suffix)

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
