import os

from tqdm import tqdm
from simulations_util import *
from collections import defaultdict
import pickle as pkl
from numpy.random import uniform
from os.path import join as oj
import sys

from imodels.tree.saps import SaplingSumRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from pygam import LinearGAM, s

sys.path.append('..')

# change working directory to project root
if os.getcwd().split('/')[-1] == 'notebooks':
    os.chdir('../..')

# choose params
n_train = np.ceil(np.geomspace(100, 2500, 8)).astype(int)  # [100, 250, 500, 750, 1000, 1500, 2000, 2500]
n_test = 500
d = 50
m = 5  # number of interaction terms
r = 3  # order of interaction
sigma = 0.1
n_avg = 10
seed = 1
thres = 5
tau = 0.5
beta = 1
sparsity = m*r

gam_terms = sum((s(i) for i in range(d)), s(0))

model_dict = {"GAM": LinearGAM(gam_terms),
              "CART": DecisionTreeRegressor(min_samples_leaf=5),
              "RF": RandomForestRegressor(),
              "SAPS": SaplingSumRegressor(),
              "XGB": XGBRegressor()}

sim_dict = {"LSS": lss_model,
            "poly": sum_of_polys,
            "linear": linear_model}


def get_sim_results(model_dict, y_gen, n_train, n_test, d, n_avg, seed, sim_name, sigma, **kwargs):
    np.random.seed(seed)
    scores = defaultdict(list)
    error_bars = defaultdict(list)
    out_dir = oj('results', sim_name)
    fname = oj('results', sim_name, 'scores.pkl')

    for n in tqdm(n_train):
        scores_n = defaultdict(list)
        for j in range(n_avg):
            X_train = uniform(low=0, high=1.0, size=(n, d))
            X_test = uniform(low=0, high=1.0, size=(n_test, d))
            y_train = y_gen(X_train, sigma, **kwargs)
            y_test = y_gen(X_test, 0, **kwargs)

            for k, m in model_dict.items():
                if k == "SAPS":
                    m.fit(X_train, y_train, min_impurity_decrease=thres*sigma**2)
                else:
                    m.fit(X_train, y_train)
                preds = m.predict(X_test)
                scores_n[k].append(mean_squared_error(y_test, preds))

        for k in scores_n:
            scores[k].append(np.mean(scores_n[k]))
            error_bars[k].append(np.std(scores_n[k]))

    os.makedirs(out_dir, exist_ok=True)
    with open(fname, 'wb') as f:
        pkl.dump((scores, error_bars), f)

    return scores, error_bars


def get_saps_trees_sim(y_gen, d, n, n_avg, seed, sim_name, sigma, **kwargs):

    np.random.seed(seed)
    out_dir = oj('results', sim_name)
    fname = oj('results', sim_name, 'trees.pkl')
    trees = []

    for j in range(n_avg):
        X_train = uniform(low=0, high=1.0, size=(n, d))
        y_train = y_gen(X_train, sigma, **kwargs)

        model = SaplingSumRegressor()
        model.fit(X_train, y_train, min_impurity_decrease=thres*sigma**2)
        trees += model.trees_

    os.makedirs(out_dir, exist_ok=True)
    with open(fname, 'wb') as f:
        pkl.dump(trees, f)


# get_sim_results(model_dict, lss_model, n_train, n_test, d, n_avg, seed, "LSS", sigma, m=m, r=r, tau=0.5)
# get_sim_results(model_dict, sum_of_polys, n_train, n_test, d, n_avg, seed, "poly", sigma, m=m, r=r)
# get_sim_results(model_dict, linear_model, n_train, n_test, d, n_avg, seed, "linear", sigma, beta=beta, s=sparsity)

get_sim_results(model_dict, lss_model, n_train, n_test, d, n_avg, seed, "single_interaction", sigma, m=1, r=8, tau=0.1)

#get_saps_trees_sim(lss_model, d, 2500, n_avg, seed, "LSS", sigma, m=m, r=r, tau=0.5)
#get_saps_trees_sim(sum_of_polys, d, 2500, n_avg, seed, "poly", sigma, m=m, r=r)