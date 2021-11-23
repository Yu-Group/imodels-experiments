import os
import pickle as pkl
import sys
from math import ceil

import matplotlib as mpl
from tqdm import tqdm

sys.path.append('../..')

mpl.rcParams['figure.dpi'] = 250
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

# change working directory to project root
if os.getcwd().split('/')[-1] == 'notebooks':
    os.chdir('../..')
from util import DATASET_PATH
from viz import *
import viz

RESULTS_PATH = oj(os.path.dirname(DATASET_PATH), 'results')

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def make_comparison_grid(metric='rocauc', num_dsets=7, datasets=[],
                         models_to_include=['SAPS', 'CART'], save_name='fig', data_type='reg_data'):
    R, C = ceil(num_dsets / 3), 3
    plt.figure(figsize=(3 * C, 2.5 * R), facecolor='w')

    COLORS = {
        'CART_(MSE)': 'orange',
        'ShrunkCART': 'orange',
        'Random_Forest': cg,
        'Shrunk_Random_Forest': cg,
        'Gradient_Boosting': cb,
        'Shrunk_Gradient_Boosting': cb,

        'Rulefit': 'green',
        'CART': 'orange',  # cp
        'C45': cb,
        'SAPS': 'black',
        'CART_(MAE)': cp,
        'SAPS_(Reweighted)': cg,
        'SAPS_(Include_Linear)': cb,

    }

    for i, dset in enumerate(tqdm(datasets[::-1][:num_dsets])):
        # try:
        dset_name = dset[0]
        #         try:
        ax = plt.subplot(R, C, i + 1)
        pkl_file = oj(RESULTS_PATH, data_type, dset_name, 'train-test/combined.pkl')
        df = pkl.load(open(pkl_file, 'rb'))['df']
        df['model'] = df.index
        for _, (name, g) in enumerate(df.groupby('model')):
            if name in models_to_include:
                x = g[f'{dset_name}_complexity']
                args = np.argsort(x)
                shrunk = 'shrunk' in name.lower()
                alpha = 1.0 if shrunk else 0.35
                lw = 1.5 if shrunk else 3
                ls = '--' if shrunk else '-'
                label = name.replace('_', ' ').replace('C45', 'C4.5').replace('Random Forest', 'RF').replace('Gradient Boosting', 'GB')\
                    if not shrunk else None
                kwargs = dict(color=COLORS.get(name, cb2), alpha=alpha, lw=lw, ls=ls, ms=10, label=label)
                #                 print(g.keys())
                x = x[args].values  # args #
                y = g[f'{dset_name}_{metric}_test'][args].values
                # print('x', x)
                # print('y', y)
                plt.plot(np.log10(x), y, **kwargs,)  # , zorder=-5)
                #             plt.plot(g[f'{dset_name}_complexity'][args], g[f'{dset_name}_{metric}_train'][args], '.--', **kwargs,
                #                      label=name + ' (Train)')
                plt.xlabel('Log Number of rules')
                # plt.xlim((0, 20))
                plt.ylabel(
                    dset_name.capitalize().replace('-', ' ') + ' ' + metric.upper().replace('ROC', '').replace('R2',
                                                                                                               '$R^2$'))
                # plt.xscale('log')
        if i % C == C - 1:
            # rect = patches.Rectangle((18, 0), 100, 1, linewidth=1, edgecolor='w', facecolor='w', zorder=-4)
            dvu.line_legend(fontsize=10, xoffset_spacing=0.1, adjust_text_labels=True)
            # ax.add_patch(rect)
            pass
        # except:
        #     print('skipping', dset_name)
        #     traceback.print_exc()
    viz.savefig(save_name)
