import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
import sys
from imodels import SaplingSumRegressor
from simulations_util import *
from collections import defaultdict
import pickle as pkl
from numpy.random import uniform

sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')

# change working directory to project root
# os.chdir('../..')
# print(os.listdir('.'))

from viz import *

if __name__ == '__main__':

    # choose params
    model = 'sum_of_squares'  # sum_of_squares
    n_train = [100, 250, 500, 750, 1000, 1500] #,1500,2000,2500]
    n_test = 500
    d = 50
    beta = 1
    sigma = 0.1
    sparsity = [10, 20]
    n_avg = 4
    seed = 1

    out_dir = f'results/{model}'
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # keys end up being saps, cart, rf
    scores = defaultdict(list)
    error_bar = defaultdict(list)
    np.random.seed(seed)

    # This cell's code is used to fit and predict for on linear model varying across
    # the number of training samples/sparsity
    for s_num, s in enumerate(sparsity):
        print('s_num', s_num)
        scores_s = defaultdict(list)
        error_bar_s = defaultdict(list)
        fname = oj(out_dir, f'scores_{s_num}.pkl')

        if os.path.exists(fname):
            continue

        for n in tqdm(n_train):
            scores_s_n = defaultdict(list)
            for j in range(n_avg):
                X_train = uniform(low=0, high=1.0, size=(n, d))
                X_test = uniform(low=0, high=1.0, size=(n_test, d))

                if model == 'sum_of_squares':
                    y_train = sum_of_squares(X_train, s, beta, sigma)
                    y_test = sum_of_squares(X_test, s, beta, 0)
                elif model == 'linear_model':
                    y_train = linear_model(X_train, s, beta, sigma)
                    y_test = linear_model(X_test, s, beta, 0)

                for k, m in zip(['SAPS', 'CART', 'RF'], [SaplingSumRegressor(),
                                                         DecisionTreeRegressor(min_samples_leaf=5),
                                                         RandomForestRegressor(n_estimators=100, max_features=0.33)]):
                    m.fit(X_train, y_train)
                    preds = m.predict(X_test)
                    scores_s_n[k].append(mean_squared_error(y_test, preds))


            for k in scores_s_n:
                scores_s[k].append(np.mean(scores_s_n[k]))
                error_bar_s[k].append(np.std(scores_s_n[k]))

        #save results
        for k in scores_s:
            scores[k].append(scores_s[k])
            error_bar[k].append(error_bar_s[k])

        os.makedirs(out_dir, exist_ok=True)
        with open(fname, 'wb') as f:
            pkl.dump((scores, error_bar), f)