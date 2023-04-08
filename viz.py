import math
import os.path
import pickle as pkl
import subprocess
from math import ceil
from os.path import dirname
from os.path import join as oj
from typing import List, Dict, Any, Union, Tuple
import warnings
from copy import deepcopy

# import adjustText
import dvu
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from config.figs_interactions.datasets import DATASETS_REGRESSION
from util import remove_x_axis_duplicates, merge_overlapping_curves

dvu.set_style()
mpl.rcParams['figure.dpi'] = 250

cb2 = '#66ccff'
cb = '#1f77b4'
cr = '#cc0000'
cp = '#cc3399'
cy = '#d8b365'
cg = '#5ab4ac'

DIR_FIGS = oj(dirname(os.path.realpath(__file__)), 'figures')
DSET_METADATA = {'sonar': (208, 60), 'heart': (270, 15), 'breast-cancer': (277, 17), 'haberman': (306, 3),
                 'ionosphere': (351, 34), 'diabetes': (768, 8), 'german-credit': (1000, 20), 'juvenile': (3640, 286),
                 'recidivism': (6172, 20), 'credit': (30000, 33), 'readmission': (101763, 150), 'friedman1': (200, 10),
                 'friedman2': (200, 4), 'friedman3': (200, 4), 'abalone': (4177, 8), 'diabetes-regr': (442, 10),
                 'california-housing': (20640, 8), 'satellite-image': (6435, 36), 'echo-months': (17496, 9),
                 'breast-tumor': (116640, 9), "vo_pati": (100, 100), "radchenko_james": (300, 50),
                 'tbi-pecarn': (42428, 121), 'csi-pecarn': (3313, 36), 'iai-pecarn': (12044, 58),
                 }


def plot_comparisons(metric='rocauc', datasets=[],
                     models_to_include=['FIGS', 'CART'],
                     models_to_include_dashed=[],
                     config_name='figs',
                     color_legend=True,
                     seed=None,
                     eps_legend_sep=0.01,
                     save_name='fig',
                     show_train=False,
                     xlim=20,
                     C=3):
    """Plots curves for different models as a function of complexity
    Note: for best legends, pass models_to_include from top to bottom

    Params
    ------
    metric: str
        Which metric to plot on y axis
    
    """
    R = ceil(len(datasets) / C)
    plt.figure(figsize=(3 * C, 2.5 * R), facecolor='w')

    COLORS = {
        'FIGS': 'black',
        'CART': 'orange',  # cp,
        'Rulefit': 'green',
        'C45': cb,
        'CART_(MSE)': 'orange',
        'CART_(MAE)': cg,
        'FIGS_(Reweighted)': cg,
        'FIGS_(Include_Linear)': cb,
        'GBDT-1': cp,
        'GBDT-2': 'gray',
        'Dist-GB-FIGS': cg,
        'Dist-RF-FIGS': cp,
        'Dist-RF-FIGS-3': 'green',
        'RandomForest': 'gray',
        'GBDT': 'black',
        'BFIGS': 'green',
        'TAO': cb,
    }

    for i, dset in enumerate(tqdm(datasets)):
        if isinstance(dset, str):
            dset_name = dset
        elif isinstance(dset, tuple):
            dset_name = dset[0]
        #         try:
        ax = plt.subplot(R, C, i + 1)
        plt.title(dset_name.capitalize().replace('-', ' ') + f' ($n={DSET_METADATA.get(dset_name, (-1))[0]}$)',
                  fontsize='medium')

        suffix = '_mean'
        if seed is None:
            pkl_file = oj('results', config_name, dset_name, 'train-test/results_aggregated.pkl')
            df = pkl.load(open(pkl_file, 'rb'))['df_mean']
            # print('ks', df.keys())
        else:
            pkl_file = oj('results', config_name, dset_name, 'train-test/seed0/results_aggregated.pkl')
            df = pkl.load(open(pkl_file, 'rb'))['df']
            suffix = ''
        texts = []
        ys = []
        for name in models_to_include_dashed + models_to_include:
            try:
                g = df.groupby('estimator').get_group(name)
            except:
                # raise UserWarning(f'tried {name} but valid keys are {df.groupby("estimator").groups.keys()}')
                print(f'tried {name} but valid keys are {df.groupby("estimator").groups.keys()}')
                continue
            #             print(g)
            #             .get_group(name)
            #         for _, (name, g) in enumerate(df.groupby('estimator')):
            #             if name in models_to_include + models_to_include_dashed:
            # print('g keys', g.keys())
            x = g['complexity' + suffix].values
            y = g[f'{metric}_test' + suffix].values
            yerr = g[f'{metric}_test' + '_std'].values
            args = np.argsort(x)
            if name in ['FIGS', 'TAO']:
                alpha = 1.0
                lw = 2
            else:
                alpha = 0.35
                lw = 1.5
            name_lab = (
                name.replace('_', ' ')
                    .replace('C45', 'C4.5')
                    .replace('GBDT-1', 'Boosted Stumps')
            )
            kwargs = dict(color=COLORS.get(name, 'gray'),
                          alpha=alpha,
                          lw=lw,
                          zorder=-5,
                          )

            #             if name == 'Dist-GB-FIGS':
            #                 print(g)
            #                 print(g.keys())
            #                 plt.plot(x[args], y[args], '.-', **kwargs)

            def select_y(y, ys, eps_legend_sep, delta=0.005):
                """Select y that doesn't overlap with previous ys by pushing things down
                """
                min_dist = 0
                while min_dist < eps_legend_sep:
                    min_dist = 1e10
                    for yy in ys:
                        if np.abs(y - yy) < min_dist:
                            min_dist = np.abs(y - yy)
                    if min_dist < eps_legend_sep:
                        y = y - delta
                return y

            if name in models_to_include:
                plt.errorbar(x[args], y[args], yerr=yerr[args], fmt='.-',
                             label=name_lab, **kwargs)
                if color_legend and i % C == C - 1 and i / C < 1:  # top-right
                    arg_rightmost_less_than_xlim = np.sum(x < xlim)
                    y = select_y(y[args][arg_rightmost_less_than_xlim],  # - eps_legend_sep / 2,
                                 ys, eps_legend_sep)
                    ys.append(y)
                    texts.append(plt.text(xlim, y,
                                          name_lab,
                                          color=COLORS.get(name, 'gray'),
                                          fontsize='medium'))

            elif name in models_to_include_dashed:
                assert x.size == 1, 'Dashed models should only have 1 complexity value!'
                plt.axhline(y[args], **kwargs, linestyle='--')
                if i % C == C - 1 and i / C < 1:  # top-right
                    ys.append(y[args])
                    texts.append(plt.text(xlim, y[args],
                                          name.replace('RandomForest', 'Random Forest').replace('GBDT', 'GBDT'),
                                          color=COLORS.get(name, 'gray'), fontsize='medium'))
            if show_train:
                plt.plot(g[f'complexity_train'][args], g[f'{dset_name}_{metric}_train'][args], '.--', **kwargs,
                         label=name + ' (Train)')
            plt.xlabel('Number of splits')
            if xlim is not None:
                plt.xlim((0, xlim))
        #         if i % C == C - 1:
        if i % C == 0:  # left col
            plt.ylabel(metric.upper()
                       .replace('ROC', '')
                       .replace('R2', '$R^2$')
                       )
        if i % C == C - 1 and i / C < 1:  # top-right
            if color_legend:
                ax.set_xlim(right=xlim * 1.5)
                #             adjustText.adjust_text(texts, only_move={'points':'', 'text':'y', 'objects':''},
                #                                    avoid_points=False, va='center')
                rect = patches.Rectangle((xlim, 0), 100, 1, linewidth=1, edgecolor='w', facecolor='w', zorder=-4)
                ylim = ax.get_ylim()
                #             dvu.line_legend(fontsize=10, xoffset_spacing=0.1, adjust_text_labels=True)
                ax.add_patch(rect)
                ax.set_ylim(ylim)
            else:
                plt.legend()
    savefig(save_name)


def plot_bests(metric='rocauc', datasets=[],
               models_to_include=['FIGS', 'CART'],
               models_to_include_dashed=[],
               config_name='figs',
               color_legend=True,
               seed=None,
               eps_legend_sep=0.01,
               plot=True,
               save_name='fig', show_train=False, xlim=20):
    """Plot bests for different models as a function of complexity
    Note: for best legends, pass models_to_include from top to bottom

    Params
    ------
    metric: str
        Which metric to plot on y axis

    """
    R, C = ceil(len(datasets) / 3), 3
    plt.figure(figsize=(3 * C, 2.5 * R), facecolor='w')

    COLORS = {  # cg, cp, cb, cp
        'FIGS': cb,
        'RFFIGS-10': '#0033cc',
        #         'RFFIGS': '#0033cc',
        #         'RFFIGS-depth4': cp,
        #         'RFFIGS-log2': 'blue',
        'CART': 'orange',
        'RF': '#ff6600',
    }

    results_all = []
    # iterate over datasets
    for i, dset in enumerate(tqdm(datasets)):
        if isinstance(dset, str):
            dset_name = dset
        elif isinstance(dset, tuple):
            dset_name = dset[0]
        ax = plt.subplot(R, C, i + 1)
        suffix = '_mean'
        if seed is None:
            pkl_file = oj('results', config_name, dset_name, 'train-test/results_aggregated.pkl')
            df = pkl.load(open(pkl_file, 'rb'))['df_mean']
        else:
            pkl_file = oj('results', config_name, dset_name, 'train-test/seed0/results_aggregated.pkl')
            df = pkl.load(open(pkl_file, 'rb'))['df']
            suffix = ''

        # iterate over models
        vals = []
        names = []
        errs = []
        results = {
            'dset': dset_name,
        }
        for name in models_to_include:
            try:
                g = df.groupby('estimator').get_group(name)
            except:
                warnings.warn(f'tried {name} but valid keys are {df.groupby("estimator").groups.keys()}')
                #                 raise Exception(f'tried {name} but valid keys are {df.groupby("estimator").groups.keys()}')
                continue

            x = g['complexity' + suffix].values
            y = g[f'{metric}_test' + suffix].values[0]
            yerr = g[f'{metric}_test' + '_std'].values[0]
            name = name.replace('RandomForest', 'RF')
            
            vals.append(y)
            errs.append(yerr)
            names.append(name)
            results.update({
                'complexity': x,
                f'{metric}_test': y,
                f'{metric}_test_std': yerr,
                'name': name,
            })
            results_all.append(deepcopy(results))
        
        if plot:
            plt.bar(names, vals,
                    yerr=yerr,
                    color=[COLORS.get(name, 'grey') for name in names])
            #         plt.grid(zorder=100000)

            plt.xticks(rotation=20)
            #         plt.bar(np.arange(len(vals)), vals)

            # plot editing
            plt.title(dset_name.capitalize().replace('-', ' ') + f' ($n={DSET_METADATA.get(dset_name, (-1))[0]}$)',
                      fontsize='medium')
            if i % C == 0:  # left col
                plt.ylabel(metric.upper()
                           .replace('ROC', '')
                           .replace('R2', '$R^2$')
                           )
            if metric.upper() == 'ROCAUC':
                plt.ylim(bottom=0.5)
        
    #         plt.legend()
    savefig(save_name)
    return results_all


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


def viz_interactions(datasets=[],
                     models_to_include=[],
                     config_name='figs_interactions',
                     save_path="") -> None:
    import dataframe_image as dfi
    for i, dset in enumerate(tqdm(datasets)):
        if isinstance(dset, str):
            dset_name = dset
        elif isinstance(dset, tuple):
            dset_name = dset[0]

        pkl_file = oj('results', config_name, dset_name, 'train-test/results_aggregated.pkl')
        df = pkl.load(open(pkl_file, 'rb'))['df_mean']
        idx = [e in models_to_include for e in df.estimator]
        estimators = df.estimator
        data_print = {}
        df_models = df.iloc[idx,].round(2)

        for est in estimators:
            data_print[est] = {}
            for kind in ["interaction"]:
                for met in ["fpr", "tpr"]:
                    tpr_mean_est = df_models.loc[df.estimator == est, f"{kind}_{met}_test_mean"].values[0]
                    tpr_std_est = df_models.loc[df.estimator == est, f"{kind}_{met}_test_std"].values[0]

                    tpr_str = f"{tpr_mean_est} ({tpr_std_est})"
                    data_print[est][f"{met.upper()} {kind.capitalize()}"] = tpr_str

        data = pd.DataFrame(data_print)
        dfi.export(data, os.path.join(save_path, f'{dset_name}.png'))


if __name__ == '__main__':
    pth = "/accounts/campus/omer_ronen/projects/tree_shrink/imodels-experiments"
    viz_interactions(DATASETS_REGRESSION,
                     ["BFIGS", "DT", "FIGS", "GB", "RF"],
                     save_path=f"{pth}/results/figs_interactions")
