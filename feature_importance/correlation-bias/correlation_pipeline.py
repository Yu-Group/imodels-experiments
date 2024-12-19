# imports from imodels
from imodels import get_clean_dataset
from imodels.tree.rf_plus.rf_plus.rf_plus_models import \
    RandomForestPlusRegressor, RandomForestPlusClassifier
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import \
    RFPlusMDI, AloRFPlusMDI
from simulations_util import partial_linear_lss_model

# imports from sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, \
    accuracy_score, r2_score, f1_score, log_loss, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# parallelization imports
from joblib import Parallel, delayed

# timing imports
import time

# other data science imports
import numpy as np
import pandas as pd
import shap
from ucimlrepo import fetch_ucirepo

# i/o imports 
import argparse
import os
from os.path import join as oj

# global variable for classification/regression status
TASK = None

def simulate_data(rho, pve, seed):
    
    np.random.seed(seed)
    
    n = 250 # number of samples
    p1 = 50  # number of correlated features
    p2 = 50  # number of uncorrelated features

    # create the covariance matrix for the first block (correlated features)
    Sigma_1 = np.full((p1, p1), rho)  # matrix filled with rho
    np.fill_diagonal(Sigma_1, 1)  # set diagonal elements to 1

    # create the covariance matrix for the second block (uncorrelated features)
    Sigma_2 = np.eye(p2)  # identity matrix for independent features

    # create the full covariance matrix by combining the two blocks
    # using np.zeros to initialize the off-diagonal blocks
    Sigma = np.block([
        [Sigma_1, np.zeros((p1, p2))],  # Correlated features with independent features (zero covariance)
        [np.zeros((p2, p1)), Sigma_2]   # Independent features (identity covariance)
    ])
    
    # mean vector (zero mean)
    mu = np.zeros(100)

    # draw X from the multivariate normal distribution
    X = np.random.multivariate_normal(mu, Sigma, size = n)
    
    y = partial_linear_lss_model(X=X, s=2, m=3, r=2, tau=0, beta=1, heritability=pve)
    
    return X, y

def split_data(X, y, test_size, seed):
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=seed)
    return X_train, X_test, y_train, y_test

def fit_models(X_train, y_train):
    # fit models
    if TASK == "classification":
        rf = RandomForestClassifier(n_estimators = 100, min_samples_leaf=3,
                                    max_features = "sqrt", random_state=42)
        rf.fit(X_train, y_train)
        rf_plus = RandomForestPlusClassifier(rf_model=rf)
        rf_plus.fit(X_train, y_train)
    elif TASK == "regression":
        rf = RandomForestRegressor(n_estimators = 100, min_samples_leaf=5,
                                   max_features = 0.33, random_state=42)
        rf.fit(X_train, y_train)
        rf_plus = RandomForestPlusRegressor(rf_model=rf)
        rf_plus.fit(X_train, y_train)
    else:
        raise ValueError("Task must be either 'classification' or 'regression'.")
    return rf, rf_plus

def get_shap(X, shap_explainer):
    if TASK == "classification":
        # the shap values are an array of shape
        # (# of samples, # of features, # of classes), and in this binary
        # classification case, we want the shap values for the positive class.
        # check_additivity=False is used to speed up computation.
        shap_values = \
            shap_explainer.shap_values(X, check_additivity=False)[:, :, 1]
    else:
        # check_additivity=False is used to speed up computation.
        shap_values = shap_explainer.shap_values(X, check_additivity=False)
    # get the rankings of the shap values. negative absolute value is taken
    # because np.argsort sorts from smallest to largest.
    shap_rankings = np.argsort(-np.abs(shap_values), axis = 1)
    return shap_values, shap_rankings

def get_lmdi(X, y, lmdi_explainer, l2norm, sign, normalize, leaf_average, ranking=False):
    # get feature importances
    lmdi = lmdi_explainer.explain_linear_partial(X, y, l2norm=l2norm, sign=sign,
                                                 normalize=normalize,
                                                 leaf_average=leaf_average,
                                                 ranking=ranking)
    mdi_rankings = lmdi_explainer.get_rankings(np.abs(lmdi))
    return lmdi, mdi_rankings

if __name__ == '__main__':
    
    # start time
    start = time.time()
    
    TASK = "regression"
    
    # store command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--rho', type=float, default=None)
    parser.add_argument('--pve', type=float, default=None)
    parser.add_argument('--njobs', type=int, default=1)
    args = parser.parse_args()
    
    # convert namespace to a dictionary
    args_dict = vars(args)

    # assign the arguments to variables
    seed = args_dict['seed']
    rho = args_dict['rho']
    pve = args_dict['pve']
    njobs = args_dict['njobs']
    
    X_train, y_train = simulate_data(rho, pve, seed)
    
    # end time
    end = time.time()
    
    # print progress message
    print(f"Progress Message 1/5: Obtained data with PVE = {pve}, rho = {rho}, and seed = {seed}.")
    print(f"Step #1 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # fit the prediction models
    rf, rf_plus = fit_models(X_train, y_train)
    
    # fit baseline model
    rf_plus_baseline = RandomForestPlusRegressor(rf_model=rf,
                                        include_raw=False, fit_on="inbag",
                                        prediction_model=LinearRegression())
    rf_plus_baseline.fit(X_train, y_train)
    
    # end time
    end = time.time()
    
    print(f"Progress Message 2/5: RF and RF+ models fit.")
    print(f"Step #2 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # obtain shap feature importances
    # shap_explainer = shap.TreeExplainer(rf)
    # shap_values, shap_rankings = get_shap(X_train, shap_explainer)
    
    # end time
    end = time.time()
    
    print(f"Progress Message 3/5: SHAP values/rankings obtained.")
    print(f"Step #3 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # obtain lmdi feature importances
    # lmdi_explainer_signed_normalized_l2_avg = AloRFPlusMDI(rf_plus, mode = "only_k")
    # lmdi_explainer_signed_normalized_l2_noavg = AloRFPlusMDI(rf_plus, mode = "only_k")
    # lmdi_explainer_signed_nonnormalized_l2_avg = AloRFPlusMDI(rf_plus, mode = "only_k")
    # lmdi_explainer_signed_nonnormalized_l2_noavg = AloRFPlusMDI(rf_plus, mode = "only_k")
    # lmdi_explainer_nonl2_avg = AloRFPlusMDI(rf_plus, mode = "only_k")
    # lmdi_explainer_nonl2_noavg = AloRFPlusMDI(rf_plus, mode = "only_k")
    # lmdi_explainer_l2_ranking = AloRFPlusMDI(rf_plus, mode = "only_k")
    # lmdi_explainer_nonl2_ranking = AloRFPlusMDI(rf_plus, mode = "only_k")
    # lmdi_explainer_normalized_l2_ranking = AloRFPlusMDI(rf_plus, mode = "only_k")
    # lmdi_explainer_nonnormalized_l2_ranking = AloRFPlusMDI(rf_plus, mode = "only_k")
    # lmdi_baseline_explainer = RFPlusMDI(rf_plus_baseline, mode = "only_k", evaluate_on = "inbag")
    # lmdi_values_signed_normalized_l2_avg, \
    #     lmdi_rankings_signed_normalized_l2_avg = \
    #         get_lmdi(X_train, y_train, lmdi_explainer_signed_normalized_l2_avg,
    #                  l2norm=True, sign=True, normalize=True, leaf_average=True)
    # lmdi_values_signed_normalized_l2_noavg, \
    #     lmdi_rankings_signed_normalized_l2_noavg = \
    #         get_lmdi(X_train, y_train,lmdi_explainer_signed_normalized_l2_noavg,
    #                  l2norm=True, sign=True, normalize=True, leaf_average=False)
    # lmdi_values_signed_nonnormalized_l2_avg, \
    #     lmdi_rankings_signed_nonnormalized_l2_avg = \
    #         get_lmdi(X_train,y_train,lmdi_explainer_signed_nonnormalized_l2_avg,
    #                  l2norm=True, sign=True, normalize=False, leaf_average=True)
    # lmdi_values_signed_nonnormalized_l2_noavg, \
    #     lmdi_rankings_signed_nonnormalized_l2_noavg = \
    #         get_lmdi(X_train, y_train,
    #                  lmdi_explainer_signed_nonnormalized_l2_noavg, l2norm=True,
    #                  sign=True, normalize=False, leaf_average=False)
    # lmdi_values_nonl2_avg, lmdi_rankings_nonl2_avg = \
    #     get_lmdi(X_train, y_train, lmdi_explainer_nonl2_avg, l2norm=False,
    #              sign=False, normalize=False, leaf_average=True)
    # lmdi_values_nonl2_noavg, lmdi_rankings_nonl2_noavg = \
    #     get_lmdi(X_train, y_train, lmdi_explainer_nonl2_noavg, l2norm=False,
    #              sign=False, normalize=False, leaf_average=False)
    # lmdi_values_l2_ranking, lmdi_rankings_l2_ranking = \
    #     get_lmdi(X_train, y_train, lmdi_explainer_l2_ranking, l2norm=True,
    #              sign=False, normalize=False, leaf_average=False, ranking=True)
    # lmdi_values_nonl2_ranking, lmdi_rankings_nonl2_ranking = \
    #     get_lmdi(X_train, y_train, lmdi_explainer_nonl2_ranking, l2norm=False,
    #                 sign=False, normalize=False, leaf_average=False, ranking=True)
    # lmdi_values_normalized_l2_ranking, lmdi_rankings_normalized_l2_ranking = \
    #     get_lmdi(X_train, y_train, lmdi_explainer_normalized_l2_ranking, l2norm=True,
    #                 sign=False, normalize=True, leaf_average=False, ranking=True)
    # lmdi_values_baseline, lmdi_rankings_baseline = \
    #     get_lmdi(X_train, y_train, lmdi_baseline_explainer, l2norm=False,
    #              sign=False, normalize=False, leaf_average=False)

    # # create storage for iteration purposes
    # lfi_values = \
    #     {'shap': shap_values,
    #      'signed_normalized_l2_avg': lmdi_values_signed_normalized_l2_avg,
    #      'signed_normalized_l2_noavg': lmdi_values_signed_normalized_l2_noavg,
    #      'signed_nonnormalized_l2_avg': lmdi_values_signed_nonnormalized_l2_avg,
    #      'signed_nonnormalized_l2_noavg':
    #          lmdi_values_signed_nonnormalized_l2_noavg,
    #      'nonl2_avg': lmdi_values_nonl2_avg,
    #      'nonl2_noavg': lmdi_values_nonl2_noavg,
    #      'l2_ranking': lmdi_values_l2_ranking,
    #      'nonl2_ranking': lmdi_values_nonl2_ranking,
    #      'normalized_l2_ranking': lmdi_values_normalized_l2_ranking,
    #      'baseline': lmdi_values_baseline}
    # lfi_rankings = \
    #     {'shap': shap_rankings,
    #      'signed_normalized_l2_avg': lmdi_rankings_signed_normalized_l2_avg,
    #      'signed_normalized_l2_noavg': lmdi_rankings_signed_normalized_l2_noavg,
    #      'signed_nonnormalized_l2_avg': lmdi_rankings_signed_nonnormalized_l2_avg,
    #      'signed_nonnormalized_l2_noavg':
    #          lmdi_rankings_signed_nonnormalized_l2_noavg,
    #      'nonl2_avg': lmdi_rankings_nonl2_avg,
    #      'nonl2_noavg': lmdi_rankings_nonl2_noavg,
    #      'l2_ranking': lmdi_rankings_l2_ranking,
    #      'nonl2_ranking': lmdi_rankings_nonl2_ranking,
    #      'normalized_l2_ranking': lmdi_values_normalized_l2_ranking,
    #      'baseline': lmdi_rankings_baseline}
    
    # get mdi importances from rf
    mdi_values = rf.feature_importances_
    mdi_rankings = np.argsort(-np.abs(mdi_values))
    
    # create storage for iteration purposes
    lfi_values = \
        {'mdi': mdi_values}
        
    lfi_rankings = \
        {'mdi': mdi_rankings}
        
    # end time
    end = time.time()
    
    print(f"Progress Message 4/5: LMDI+ values/rankings obtained.")
    print(f"Step #4 took {end-start} seconds.")
    
    result_dir = oj(os.path.dirname(os.path.realpath(__file__)),
                    f'results/pve{pve}/rho{rho}/seed{seed}')
    
    # get result dataframes
    for method, values in lfi_values.items():
        df = pd.DataFrame(values)
        directory = oj(result_dir,
                     f'values')
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(oj(directory, f'{method}.csv'),
                  index=False)
    for method, rankings in lfi_rankings.items():
        df = pd.DataFrame(rankings)
        directory = oj(result_dir,
                     f'rankings')
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(oj(directory, f'{method}.csv'),
                  index=False)
        
    # end time
    end = time.time()
        
    print(f"Progress Message 5/5: Results saved to {result_dir}.")
    print(f"Step #5 took {end-start} seconds.")