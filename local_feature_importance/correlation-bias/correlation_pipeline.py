# imports from imodels
from imodels.tree.rf_plus.rf_plus.rf_plus_models import \
    RandomForestPlusRegressor, RandomForestPlusClassifier
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import LMDIPlus
from simulations_util import partial_linear_lss_model

# imports from sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV

# timing imports
import time

# other data science imports
import numpy as np
import pandas as pd
import shap
import lime

# i/o imports 
import argparse
import os
from os.path import join as oj

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
    
    rf = RandomForestRegressor(n_estimators = 100, min_samples_leaf=5,
                                max_features = 0.33, random_state=42)
    rf.fit(X_train, y_train)
    
    # baseline rf+
    rf_plus_baseline = RandomForestPlusRegressor(rf_model=rf,
                                    include_raw=False, fit_on="inbag",
                                    prediction_model=LinearRegression())
    rf_plus_baseline.fit(X_train, y_train)
    
    # elastic net rf+
    rf_plus_elastic = RandomForestPlusRegressor(rf_model=rf,
                                        prediction_model=ElasticNetCV(cv=5,
                                    l1_ratio=[0.1,0.5,0.7,0.9,0.95,0.99],
                                    max_iter=10000,random_state=42))
    rf_plus_elastic.fit(X_train, y_train)
    
    # return rf, rf_plus_baseline, rf_plus_ridge, rf_plus_lasso, rf_plus_elastic
    return rf, rf_plus_baseline, rf_plus_elastic

def get_shap(X, shap_explainer):
    
    # check_additivity=False is used to speed up computation.
    shap_values = shap_explainer.shap_values(X, check_additivity=False)
    # get the rankings of the shap values. negative absolute value is taken
    # because np.argsort sorts from smallest to largest.
    shap_rankings = np.argsort(-np.abs(shap_values), axis = 1)
    return shap_values, shap_rankings

def get_lime(X: np.ndarray, rf):
    """
    Get the LIME values and rankings for the given data.
    
    Inputs:
    - X (np.ndarray): The feature matrix.
    - rf (RandomForestClassifier/Regressor): The fitted RF object.
    
    Outputs:
    - lime_values (np.ndarray): The LIME values.
    - lime_rankings (np.ndarray): The LIME rankings.
    """
    
    lime_values = np.zeros((X.shape[0], X.shape[1]))
    explainer = lime.lime_tabular.LimeTabularExplainer(X, verbose = False,
                                                       mode = "regression")
    num_features = X.shape[1]
    for i in range(X.shape[0]):
        exp = explainer.explain_instance(X[i, :], rf.predict,
                                         num_features = num_features)
        original_feature_importance = exp.as_map()[1]
        sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
        for j in range(num_features):
            lime_values[i, j] = sorted_feature_importance[j][1]
        
        # get the rankings of the shap values. negative absolute value is taken
        # because np.argsort sorts from smallest to largest.
        lime_rankings = np.argsort(-np.abs(lime_values), axis = 1)    
        
    return lime_values, lime_rankings

def get_lmdi(X, y, lmdi_plus_explainer, ranking):
    
    # get feature importances
    lmdi_plus = lmdi_plus_explainer.get_lmdi_plus_scores(X, y, ranking=ranking)
    
    lmdi_plus_rankings = np.argsort(-np.abs(lmdi_plus), axis = 1)
    
    return lmdi_plus, lmdi_plus_rankings

if __name__ == '__main__':
    
    # start time
    start = time.time()
        
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
    rf, rf_plus_baseline, rf_plus_elastic = fit_models(X_train, y_train)
            
    # end time
    end = time.time()
    
    print(f"Progress Message 2/5: RF/RF+ and GB/GB+ models fit.")
    print(f"Step #2 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # obtain shap feature importances
    shap_rf_explainer = shap.TreeExplainer(rf)
    shap_rf_values, shap_rf_rankings = get_shap(X_train, shap_rf_explainer)
    
    # end time
    end = time.time()
    
    print(f"Progress Message 3/5: SHAP values/rankings obtained.")
    print(f"Step #3 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # obtain LIME feature importances
    lime_rf_values, lime_rf_rankings = get_lime(X_train, rf)
    
    # end time
    end = time.time()
    
    print(f"Progress Message 4/5: LIME values/rankings obtained.")
    print(f"Step #4 took {end-start} seconds.")
    
    # start time
    start = time.time()
                
    # create the explainer objects for each variant
    lmdi_plus_rf_explainer = LMDIPlus(rf_plus_elastic, evaluate_on = "all")
    baseline_rf_explainer = LMDIPlus(rf_plus_baseline, evaluate_on = "inbag")
    
    # initialize storage mappings
    lfi_values = {}
    lfi_rankings = {}
    
    # obtain feature importances
    lmdi_plus_values, lmdi_plus_rankings = get_lmdi(X_train, y_train,
                                                  lmdi_plus_rf_explainer,
                                                  ranking=True)
    lfi_values["lmdi_plus"] = lmdi_plus_values
    lfi_rankings["lmdi_plus"] = lmdi_plus_rankings
    baseline_rf_values, baseline_rf_rankings = get_lmdi(X_train, y_train, baseline_rf_explainer,
                                                  ranking=False)
    lfi_rankings["lmdi_baseline"] = baseline_rf_rankings
    lfi_values["lmdi_baseline"] = baseline_rf_values
    lfi_rankings["shap"] = shap_rf_rankings
    lfi_values["shap"] = shap_rf_values
    lfi_rankings["lime"] = lime_rf_rankings
    lfi_values["lime"] = lime_rf_values
    
    # end time
    end = time.time()
    
    print(f"Progress Message 5/5: LMDI+ values/rankings obtained.")
    print(f"Step #5 took {end-start} seconds.")
    
    result_dir = oj(os.path.dirname(os.path.realpath(__file__)),
                    f'results/pve{pve}/rho{rho}/seed{seed}')
    
    # get result dataframes
    for method, values in lfi_values.items():
        df = pd.DataFrame(values)
        directory = oj(result_dir, "rf",
                     f'values')
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(oj(directory, f'{method}.csv'),
                  index=False)
    for method, rankings in lfi_rankings.items():
        df = pd.DataFrame(rankings)
        directory = oj(result_dir, "rf",
                     f'rankings')
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(oj(directory, f'{method}.csv'),
                  index=False)
    
    # end time
    end = time.time()
        
    print(f"Results saved to {result_dir}.")