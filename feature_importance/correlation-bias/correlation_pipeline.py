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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, \
    GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, \
    RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV
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
import lime
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

    if TASK == "classification":
        rf = RandomForestClassifier(n_estimators = 100, min_samples_leaf=1,
                                    max_features = "sqrt", random_state=42)
        rf.fit(X_train, y_train)
        
        # ridge rf+
        rf_plus_ridge = RandomForestPlusClassifier(rf_model=rf,
                            prediction_model=LogisticRegressionCV(penalty='l2',
                                        cv=5, max_iter=10000, random_state=42))
        rf_plus_ridge.fit(X_train, y_train)

        # lasso rf+
        rf_plus_lasso = RandomForestPlusClassifier(rf_model=rf,
                            prediction_model=LogisticRegressionCV(penalty='l1',
                                    solver = 'saga', cv=3, n_jobs=-1, tol=5e-4,
                                    max_iter=5000, random_state=42))
        rf_plus_lasso.fit(X_train, y_train)

        # elastic net rf+
        rf_plus_elastic = RandomForestPlusClassifier(rf_model=rf,
                    prediction_model=LogisticRegressionCV(penalty='elasticnet',
                            l1_ratios=[0.1,0.5,0.9,0.99], solver='saga', cv=3,
                        n_jobs=-1, tol=5e-4, max_iter=5000, random_state=42))
        rf_plus_elastic.fit(X_train, y_train)
        
    elif TASK == "regression":
        rf = RandomForestRegressor(n_estimators = 100, min_samples_leaf=5,
                                   max_features = 0.33, random_state=42)
        rf.fit(X_train, y_train)
        
        # baseline rf+
        rf_plus_baseline = RandomForestPlusRegressor(rf_model=rf,
                                        include_raw=False, fit_on="inbag",
                                        prediction_model=LinearRegression())
        rf_plus_baseline.fit(X_train, y_train)
        
        # ridge rf+
        rf_plus_ridge = RandomForestPlusRegressor(rf_model=rf,
                                                prediction_model=RidgeCV(cv=5))
        rf_plus_ridge.fit(X_train, y_train)
        
        # lasso rf+
        rf_plus_lasso = RandomForestPlusRegressor(rf_model=rf,
                                                  prediction_model=LassoCV(cv=5,
                                            max_iter=10000, random_state=42))
        rf_plus_lasso.fit(X_train, y_train)
        
        # elastic net rf+
        rf_plus_elastic = RandomForestPlusRegressor(rf_model=rf,
                                            prediction_model=ElasticNetCV(cv=5,
                                        l1_ratio=[0.1,0.5,0.7,0.9,0.95,0.99],
                                        max_iter=10000,random_state=42))
        rf_plus_elastic.fit(X_train, y_train)
        
    else:
        raise ValueError("Task must be either 'classification' or 'regression'.")
    
    return rf, rf_plus_baseline, rf_plus_ridge, rf_plus_lasso, rf_plus_elastic

def fit_gb_models(X_train: np.ndarray, y_train: np.ndarray):
    """
    Fit the prediction models for the subgroup experiments.
    
    Inputs:
    - X_train (np.ndarray): The feature matrix for the training set.
    - y_train (np.ndarray): The target vector for the training set.
    - task (str): The task type, either 'classification' or 'regression'.
    
    Outputs:
    - rf (RandomForestClassifier/RandomForestRegressor): The RF.
    - rf_plus_baseline (RandomForestPlusClassifier/RandomForestPlusRegressor):
                        The baseline RF+ (no raw feature, only on in-bag
                        samples, regular linear/logistic prediction model).
    - rf_plus_ridge (RandomForestPlusClassifier/RandomForestPlusRegressor):
                     The RF+ with a ridge prediction model.
    - rf_plus_lasso (RandomForestPlusClassifier/RandomForestPlusRegressor):
                     The RF+ with a lasso prediction model.
    - rf_plus_elastic (RandomForestPlusClassifier/RandomForestPlusRegressor):
                       The RF+ with an elastic net prediction model.
    """
    
    # if classification, fit classifiers
    if TASK == "classification":

        # not supported, throw error
        raise ValueError("Gradient boosting not supported for classification.")
    
    # if regression, fit regressors
    elif TASK == "regression":
        
        # fit random forest with params from MDI+
        gb = GradientBoostingRegressor(random_state=42)
        gb.fit(X_train, y_train)
        
        # baseline rf+ includes no raw feature and only fits on in-bag samples
        gb_plus_baseline = RandomForestPlusRegressor(rf_model=gb,
                                        include_raw=False, fit_on="inbag",
                                        prediction_model=LinearRegression())
        gb_plus_baseline.fit(X_train, y_train, n_jobs=None)
        
        # ridge rf+
        gb_plus_ridge = RandomForestPlusRegressor(rf_model=gb,
                                                prediction_model=RidgeCV(cv=5))
        gb_plus_ridge.fit(X_train, y_train, n_jobs=None)
        
        # lasso rf+
        gb_plus_lasso = RandomForestPlusRegressor(rf_model=gb,
                                                  prediction_model=LassoCV(cv=5,
                                            max_iter=10000, random_state=42))
        gb_plus_lasso.fit(X_train, y_train, n_jobs=None)
        
        # elastic net rf+
        gb_plus_elastic = RandomForestPlusRegressor(rf_model=gb,
                                            prediction_model=ElasticNetCV(cv=5,
                                        l1_ratio=[0.1,0.5,0.7,0.9,0.95,0.99],
                                        max_iter=10000,random_state=42))
        gb_plus_elastic.fit(X_train, y_train, n_jobs=None)
    
    # otherwise, throw error
    else:
        raise ValueError("Task must be 'classification' or 'regression'.")
    
    # return tuple of models
    return gb, gb_plus_baseline, gb_plus_ridge, gb_plus_lasso, gb_plus_elastic
    

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

def get_lime(X: np.ndarray, rf, task: str):
    """
    Get the LIME values and rankings for the given data.
    
    Inputs:
    - X (np.ndarray): The feature matrix.
    - rf (RandomForestClassifier/Regressor): The fitted RF object.
    - task (str): The task type, either 'classification' or 'regression'.
    
    Outputs:
    - lime_values (np.ndarray): The LIME values.
    - lime_rankings (np.ndarray): The LIME rankings.
    """
    
    lime_values = np.zeros((X.shape[0], X.shape[1]))
    explainer = lime.lime_tabular.LimeTabularExplainer(X, verbose = False,
                                                       mode = task)
    num_features = X.shape[1]
    for i in range(X.shape[0]):
        if task == 'classification':
            exp = explainer.explain_instance(X[i, :], rf.predict_proba,
                                             num_features = num_features)
        else:
            exp = explainer.explain_instance(X[i, :], rf.predict,
                                             num_features = num_features)
        original_feature_importance = exp.as_map()[1]
        # print("----------------")
        # print("Original feature importance")
        # print(original_feature_importance)
        sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
        # print("----------------")
        # print("Sorted feature importance")
        # print(sorted_feature_importance)
        # print("----------------")
        for j in range(num_features):
            lime_values[i, j] = sorted_feature_importance[j][1]
        
        # get the rankings of the shap values. negative absolute value is taken
        # because np.argsort sorts from smallest to largest.
        lime_rankings = np.argsort(-np.abs(lime_values), axis = 1)    
        
    return lime_values, lime_rankings

def get_lmdi(X, y, lmdi_explainer, normalize, square, ranking):
    
    # get feature importances
    lmdi = lmdi_explainer.explain_linear_partial(X, y, normalize=normalize,
                                                 square = square,
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
    rf, rf_plus_baseline, rf_plus_ridge, rf_plus_lasso, rf_plus_elastic = fit_models(X_train, y_train)
    gb, gb_plus_baseline, gb_plus_ridge, gb_plus_lasso, gb_plus_elastic = fit_gb_models(X_train, y_train)
        
    # end time
    end = time.time()
    
    print(f"Progress Message 2/5: RF/RF+ and GB/GB+ models fit.")
    print(f"Step #2 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # obtain shap feature importances
    shap_rf_explainer = shap.TreeExplainer(rf)
    shap_rf_values, shap_rf_rankings = get_shap(X_train, shap_rf_explainer)
    shap_gb_explainer = shap.TreeExplainer(gb)
    shap_gb_values, shap_gb_rankings = get_shap(X_train, shap_gb_explainer)
    
    # end time
    end = time.time()
    
    print(f"Progress Message 3/5: SHAP values/rankings obtained.")
    print(f"Step #3 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # obtain LIME feature importances
    lime_rf_values, lime_rf_rankings = get_lime(X_train, rf, TASK)
    lime_gb_values, lime_gb_rankings = get_lime(X_train, gb, TASK)
    
    # end time
    end = time.time()
    
    print(f"Progress Message 4/5: LIME values/rankings obtained.")
    print(f"Step #4 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    glm = ["ridge", "lasso", "elastic"]
    normalize = {True: "normed", False: "nonnormed"}
    square = {True: "squared", False: "nosquared"}
    ranking = {True: "rank", False: "norank"}
    
    # create the mapping of variants to argument mappings
    lmdi_variants = {}
    for g in glm:
        for n in normalize:
            for s in square:
                if (not n) and (s):
                    continue
                for r in ranking:
                    # create the name the variant will be stored under
                    variant_name = f"{g}_{normalize[n]}_{square[s]}_{ranking[r]}"
                    # store the arguments for the lmdi+ explainer
                    arg_map = {"glm": g, "normalize": n, "square": s,
                               "ranking": r}
                    lmdi_variants[variant_name] = arg_map
                
    # create the explainer objects for each variant
    lmdi_rf_explainers = {}
    for variant_name in lmdi_variants.keys():
        if lmdi_variants[variant_name]["glm"] == "ridge":
            lmdi_rf_explainers[variant_name] = RFPlusMDI(rf_plus_ridge,
                                                      mode = "only_k",
                                                      evaluate_on = "all")
        elif lmdi_variants[variant_name]["glm"] == "lasso":
            lmdi_rf_explainers[variant_name] = RFPlusMDI(rf_plus_lasso,
                                                      mode = "only_k",
                                                      evaluate_on = "all")
        elif lmdi_variants[variant_name]["glm"] == "elastic":
            lmdi_rf_explainers[variant_name] = RFPlusMDI(rf_plus_elastic,
                                                      mode = "only_k",
                                                      evaluate_on = "all")
        else:
            raise ValueError("Invalid GLM type.")
    
    baseline_rf_explainer = RFPlusMDI(rf_plus_baseline, mode = "only_k",
                                   evaluate_on = "inbag")
    
    lmdi_gb_explainers = {}
    for variant_name in lmdi_variants.keys():
        if lmdi_variants[variant_name]["glm"] == "ridge":
            lmdi_gb_explainers[variant_name] = RFPlusMDI(gb_plus_ridge,
                                                      mode = "only_k",
                                                      evaluate_on = "all")
        elif lmdi_variants[variant_name]["glm"] == "lasso":
            lmdi_gb_explainers[variant_name] = RFPlusMDI(gb_plus_lasso,
                                                      mode = "only_k",
                                                      evaluate_on = "all")
        elif lmdi_variants[variant_name]["glm"] == "elastic":
            lmdi_gb_explainers[variant_name] = RFPlusMDI(gb_plus_elastic,
                                                      mode = "only_k",
                                                      evaluate_on = "all")
        else:
            raise ValueError("Invalid GLM type.")
        
    baseline_gb_explainer = RFPlusMDI(gb_plus_baseline, mode = "only_k",
                                      evaluate_on = "inbag")

    # initialize storage mappings
    lmdi_rf_values = {}
    lmdi_rf_rankings = {}
    for name, explainer in lmdi_rf_explainers.items():
        
        # skip through the baseline model, since we have already done it
        if name == "lmdi_baseline" or name == "lmdi_lasso" or name == "lmdi_ridge":
            continue
        
        # get the argument mapping
        variant_args = lmdi_variants[name]    
        
        # set the values by calling explain on the object with the args from
        # input mapping    
        lmdi_rf_values[name] = explainer.explain_linear_partial(X_train, y_train,
                                        normalize=variant_args["normalize"],
                                        square=variant_args["square"],
                                        ranking=variant_args["ranking"])
        
        # get rankings using the method in the explainer class. absolute value
        # taken to ensure that the rankings are based on the magnitude.
        lmdi_rf_rankings[name] = explainer.get_rankings(np.abs(lmdi_rf_values[name]))
    
    lmdi_gb_values = {}
    lmdi_gb_rankings = {}
    for name, explainer in lmdi_gb_explainers.items():
        
        # skip through the baseline model, since we have already done it
        if name == "lmdi_baseline" or name == "lmdi_lasso" or name == "lmdi_ridge":
            continue
        
        # get the argument mapping
        variant_args = lmdi_variants[name]    
        
        # set the values by calling explain on the object with the args from
        # input mapping    
        lmdi_gb_values[name] = explainer.explain_linear_partial(X_train, y_train,
                                        normalize=variant_args["normalize"],
                                        square=variant_args["square"],
                                        ranking=variant_args["ranking"])
        
        # get rankings using the method in the explainer class. absolute value
        # taken to ensure that the rankings are based on the magnitude.
        lmdi_gb_rankings[name] = explainer.get_rankings(np.abs(lmdi_gb_values[name]))
    
    # obtain lmdi feature importances
    baseline_rf_values, baseline_rf_rankings = get_lmdi(X_train, y_train, baseline_rf_explainer,
                                                  normalize=False, square=False, ranking=False)
    lmdi_rf_rankings["lmdi_baseline"] = baseline_rf_rankings
    lmdi_rf_values["lmdi_baseline"] = baseline_rf_values
    lmdi_rf_rankings["shap"] = shap_rf_rankings
    lmdi_rf_values["shap"] = shap_rf_values
    lmdi_rf_rankings["lime"] = lime_rf_rankings
    lmdi_rf_values["lime"] = lime_rf_values
    
    baseline_gb_values, baseline_gb_rankings = get_lmdi(X_train, y_train, baseline_gb_explainer,
                                                  normalize=False, square=False, ranking=False)
    lmdi_gb_rankings["lmdi_baseline"] = baseline_gb_rankings
    lmdi_gb_values["lmdi_baseline"] = baseline_gb_values
    lmdi_gb_rankings["shap"] = shap_gb_rankings
    lmdi_gb_values["shap"] = shap_gb_values
    lmdi_gb_rankings["lime"] = lime_gb_rankings
    lmdi_gb_values["lime"] = lime_gb_values
    
    # # get mdi importances from rf
    # mdi_values = rf.feature_importances_
    # mdi_rankings = np.argsort(-np.abs(mdi_values))
    
    # # create storage for iteration purposes
    # lmdi_values['mdi'] = mdi_values
    # lmdi_rankings['mdi'] = mdi_rankings
        
    # end time
    end = time.time()
    
    print(f"Progress Message 5/5: LMDI+ values/rankings obtained.")
    print(f"Step #5 took {end-start} seconds.")
    
    result_dir = oj(os.path.dirname(os.path.realpath(__file__)),
                    f'results/pve{pve}/rho{rho}/seed{seed}')
    
    # get result dataframes
    for method, values in lmdi_rf_values.items():
        df = pd.DataFrame(values)
        directory = oj(result_dir, "rf",
                     f'values')
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(oj(directory, f'{method}.csv'),
                  index=False)
    for method, rankings in lmdi_rf_rankings.items():
        df = pd.DataFrame(rankings)
        directory = oj(result_dir, "rf",
                     f'rankings')
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(oj(directory, f'{method}.csv'),
                  index=False)
    for method, values in lmdi_gb_values.items():
        df = pd.DataFrame(values)
        directory = oj(result_dir, "gb",
                     f'values')
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(oj(directory, f'{method}.csv'),
                  index=False)
    for method, rankings in lmdi_gb_rankings.items():
        df = pd.DataFrame(rankings)
        directory = oj(result_dir, "gb",
                     f'rankings')
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(oj(directory, f'{method}.csv'),
                  index=False)
    
    # end time
    end = time.time()
        
    print(f"Results saved to {result_dir}.")