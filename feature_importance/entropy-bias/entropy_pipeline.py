# imports from imodels
from imodels import get_clean_dataset
from imodels.tree.rf_plus.rf_plus.rf_plus_models import \
    RandomForestPlusRegressor, RandomForestPlusClassifier
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import \
    RFPlusMDI, AloRFPlusMDI

# imports from sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, \
    accuracy_score, r2_score, f1_score, log_loss, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, \
    GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, \
    RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV
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

def simulate_data(n, seed):
    np.random.seed(seed)
    
    # get X1 from Bernoulli(0.5)
    X1 = np.random.binomial(1, 0.5, size=(n,))
    # get X2 from N(0, 1)
    X2 = np.random.normal(size=(n,))
    # get X3 from uniform discrete distribution with four categories
    X3 = np.random.randint(1, 4, size=(n,))
    # get X4 from uniform discrete distribution with ten categories
    X4 = np.random.randint(1, 10, size=(n,))
    # get X5 from uniform discrete distribution with twenty categories
    X5 = np.random.randint(1, 20, size=(n,))

    # combine the features
    X = np.column_stack((X1, X2, X3, X4, X5))
    
    # create y from X1 and noise
    if TASK == "regression":
        # y = X1 + N(0, sigma^2) where sigma^2 chosen to achieve PVE = 0.1
        heritability = 0.1
        sigma = (np.var(X1) * ((1.0 - heritability) / heritability)) ** 0.5
        epsilon = np.random.randn(n)
        y = X1 + sigma * epsilon
    elif TASK == "classification":
        # y has probability (1+x1)/3 of being 1
        y = np.random.binomial(1, (1 + X1) / 3, size=(n,))
    
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

# def fit_models(X_train, y_train):
#     # fit models
#     if TASK == "classification":
#         rf = RandomForestClassifier(n_estimators = 100, min_samples_leaf=3,
#                                     max_features = "sqrt", random_state=42)
#         rf.fit(X_train, y_train)
#         rf_plus = RandomForestPlusClassifier(rf_model=rf)
#         rf_plus.fit(X_train, y_train)
#     elif TASK == "regression":
#         rf = RandomForestRegressor(n_estimators = 100, min_samples_leaf=5,
#                                    max_features = 0.33, random_state=42)
#         rf.fit(X_train, y_train)
#         rf_plus = RandomForestPlusRegressor(rf_model=rf)
#         rf_plus.fit(X_train, y_train)
#     else:
#         raise ValueError("Task must be either 'classification' or 'regression'.")
#     return rf, rf_plus

# def get_shap(X, shap_explainer):
#     if TASK == "classification":
#         # the shap values are an array of shape
#         # (# of samples, # of features, # of classes), and in this binary
#         # classification case, we want the shap values for the positive class.
#         # check_additivity=False is used to speed up computation.
#         shap_values = \
#             shap_explainer.shap_values(X, check_additivity=False)[:, :, 1]
#     else:
#         # check_additivity=False is used to speed up computation.
#         shap_values = shap_explainer.shap_values(X, check_additivity=False)
#     # get the rankings of the shap values. negative absolute value is taken
#     # because np.argsort sorts from smallest to largest.
#     shap_rankings = np.argsort(-np.abs(shap_values), axis = 1)
#     return shap_values, shap_rankings

# def get_lmdi(X, y, lmdi_explainer, l2norm, sign, normalize, leaf_average, ranking=False):
#     # get feature importances
#     lmdi = lmdi_explainer.explain_linear_partial(X, y, l2norm=l2norm, sign=sign,
#                                                  normalize=normalize,
#                                                  leaf_average=leaf_average,
#                                                  ranking=ranking)
#     mdi_rankings = lmdi_explainer.get_rankings(np.abs(lmdi))
#     return lmdi, mdi_rankings

if __name__ == '__main__':
    
    # start time
    start = time.time()
    
    # store command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--njobs', type=int, default=1)
    args = parser.parse_args()
    
    # convert namespace to a dictionary
    args_dict = vars(args)

    # assign the arguments to variables
    seed = args_dict['seed']
    n = args_dict['n']
    TASK = args_dict['task']
    use_test = bool(args_dict['test'])
    njobs = args_dict['njobs']
    
    train_size = 500
    X, y = simulate_data(500 + n, seed)
    # since data is simulated, we can split it in a deterministic way
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # end time
    end = time.time()
    
    # print progress message
    print(f"Progress Message 1/5: Obtained {TASK} data with n = {n}.")
    print(f"Step #1 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # fit the prediction models
    rf, rf_plus = fit_models(X_train, y_train)
    
    # fit baseline model
    if TASK == "classification":
        rf_plus_baseline = RandomForestPlusClassifier(rf_model=rf,
                                        include_raw=False, fit_on="inbag",
                                        prediction_model=LogisticRegression())
    elif TASK == "regression":
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
    shap_explainer = shap.TreeExplainer(rf)
    shap_values, shap_rankings = get_shap(X_train, shap_explainer)
    
    # end time
    end = time.time()
    
    print(f"Progress Message 3/5: SHAP values/rankings obtained.")
    print(f"Step #3 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # obtain lmdi feature importances
    lmdi_explainer_signed_normalized_l2_avg = AloRFPlusMDI(rf_plus, mode = "only_k")
    lmdi_explainer_signed_normalized_l2_noavg = AloRFPlusMDI(rf_plus, mode = "only_k")
    lmdi_explainer_signed_nonnormalized_l2_avg = AloRFPlusMDI(rf_plus, mode = "only_k")
    lmdi_explainer_signed_nonnormalized_l2_noavg = AloRFPlusMDI(rf_plus, mode = "only_k")
    lmdi_explainer_nonl2_avg = AloRFPlusMDI(rf_plus, mode = "only_k")
    lmdi_explainer_nonl2_noavg = AloRFPlusMDI(rf_plus, mode = "only_k")
    lmdi_explainer_l2_ranking = AloRFPlusMDI(rf_plus, mode = "only_k")
    lmdi_explainer_nonl2_ranking = AloRFPlusMDI(rf_plus, mode = "only_k")
    lmdi_explainer_normalized_l2_ranking = AloRFPlusMDI(rf_plus, mode = "only_k")
    lmdi_explainer_nonnormalized_l2_ranking = AloRFPlusMDI(rf_plus, mode = "only_k")
    lmdi_baseline_explainer = RFPlusMDI(rf_plus_baseline, mode = "only_k", evaluate_on = "inbag")
    lmdi_values_signed_normalized_l2_avg, \
        lmdi_rankings_signed_normalized_l2_avg = \
            get_lmdi(X_train, y_train, lmdi_explainer_signed_normalized_l2_avg,
                     l2norm=True, sign=True, normalize=True, leaf_average=True)
    lmdi_values_signed_normalized_l2_noavg, \
        lmdi_rankings_signed_normalized_l2_noavg = \
            get_lmdi(X_train, y_train,lmdi_explainer_signed_normalized_l2_noavg,
                     l2norm=True, sign=True, normalize=True, leaf_average=False)
    lmdi_values_signed_nonnormalized_l2_avg, \
        lmdi_rankings_signed_nonnormalized_l2_avg = \
            get_lmdi(X_train,y_train,lmdi_explainer_signed_nonnormalized_l2_avg,
                     l2norm=True, sign=True, normalize=False, leaf_average=True)
    lmdi_values_signed_nonnormalized_l2_noavg, \
        lmdi_rankings_signed_nonnormalized_l2_noavg = \
            get_lmdi(X_train, y_train,
                     lmdi_explainer_signed_nonnormalized_l2_noavg, l2norm=True,
                     sign=True, normalize=False, leaf_average=False)
    lmdi_values_nonl2_avg, lmdi_rankings_nonl2_avg = \
        get_lmdi(X_train, y_train, lmdi_explainer_nonl2_avg, l2norm=False,
                 sign=False, normalize=False, leaf_average=True)
    lmdi_values_nonl2_noavg, lmdi_rankings_nonl2_noavg = \
        get_lmdi(X_train, y_train, lmdi_explainer_nonl2_noavg, l2norm=False,
                 sign=False, normalize=False, leaf_average=False)
    lmdi_values_l2_ranking, lmdi_rankings_l2_ranking = \
        get_lmdi(X_train, y_train, lmdi_explainer_l2_ranking, l2norm=True,
                 sign=False, normalize=False, leaf_average=False, ranking=True)
    lmdi_values_nonl2_ranking, lmdi_rankings_nonl2_ranking = \
        get_lmdi(X_train, y_train, lmdi_explainer_nonl2_ranking, l2norm=False,
                    sign=False, normalize=False, leaf_average=False, ranking=True)
    lmdi_values_normalized_l2_ranking, lmdi_rankings_normalized_l2_ranking = \
        get_lmdi(X_train, y_train, lmdi_explainer_normalized_l2_ranking, l2norm=True,
                    sign=False, normalize=True, leaf_average=False, ranking=True)
    lmdi_values_baseline, lmdi_rankings_baseline = \
        get_lmdi(X_train, y_train, lmdi_baseline_explainer, l2norm=False,
                 sign=False, normalize=False, leaf_average=False)
    if use_test:
        lmdi_values_signed_normalized_l2_avg, \
            lmdi_rankings_signed_normalized_l2_avg = \
                get_lmdi(X_test, None, lmdi_explainer_signed_normalized_l2_avg, l2norm=True, sign=True,
                     normalize=True, leaf_average=True)
        lmdi_values_signed_normalized_l2_noavg, \
            lmdi_rankings_signed_normalized_l2_noavg = \
                get_lmdi(X_test, None, lmdi_explainer_signed_normalized_l2_noavg, l2norm=True, sign=True,
                        normalize=True, leaf_average=False)
        lmdi_values_signed_nonnormalized_l2_avg, \
            lmdi_rankings_signed_nonnormalized_l2_avg = \
                get_lmdi(X_test, None, lmdi_explainer_signed_nonnormalized_l2_avg, l2norm=True, sign=True,
                        normalize=False, leaf_average=True)
        lmdi_values_signed_nonnormalized_l2_noavg, \
            lmdi_rankings_signed_nonnormalized_l2_noavg = \
                get_lmdi(X_test, None, lmdi_explainer_signed_nonnormalized_l2_noavg, l2norm=True, sign=True,
                        normalize=False, leaf_average=False)
        lmdi_values_nonl2_avg, lmdi_rankings_nonl2_avg = \
            get_lmdi(X_test, None, lmdi_explainer_nonl2_avg, l2norm=False, sign=False,
                    normalize=False, leaf_average=True)
        lmdi_values_nonl2_noavg, lmdi_rankings_nonl2_noavg = \
            get_lmdi(X_test, None, lmdi_explainer_nonl2_noavg, l2norm=False, sign=False,
                    normalize=False, leaf_average=False)
        lmdi_values_l2_ranking, lmdi_rankings_l2_ranking = \
            get_lmdi(X_test, None, lmdi_explainer_l2_ranking, l2norm=True,
                    sign=False, normalize=False, leaf_average=False, ranking=True)
        lmdi_values_nonl2_ranking, lmdi_rankings_nonl2_ranking = \
            get_lmdi(X_test, None, lmdi_explainer_nonl2_ranking, l2norm=False,
                        sign=False, normalize=False, leaf_average=False, ranking=True)
        lmdi_values_normalized_l2_ranking, lmdi_rankings_normalized_l2_ranking = \
            get_lmdi(X_test, None, lmdi_explainer_l2_ranking, l2norm=True,
                    sign=False, normalize=True, leaf_average=False, ranking=True)
        lmdi_values_baseline, lmdi_rankings_baseline = \
            get_lmdi(X_test, None, lmdi_baseline_explainer, l2norm=True, sign=False,
                    normalize=False, leaf_average=False)

    # get mdi importances from rf
    mdi_values = rf.feature_importances_
    mdi_rankings = np.argsort(-np.abs(mdi_values))

    # create storage for iteration purposes
    lfi_values = \
        {'shap': shap_values,
         'signed_normalized_l2_avg': lmdi_values_signed_normalized_l2_avg,
         'signed_normalized_l2_noavg': lmdi_values_signed_normalized_l2_noavg,
         'signed_nonnormalized_l2_avg': lmdi_values_signed_nonnormalized_l2_avg,
         'signed_nonnormalized_l2_noavg':
             lmdi_values_signed_nonnormalized_l2_noavg,
         'nonl2_avg': lmdi_values_nonl2_avg,
         'nonl2_noavg': lmdi_values_nonl2_noavg,
         'l2_ranking': lmdi_values_l2_ranking,
         'nonl2_ranking': lmdi_values_nonl2_ranking,
         'normalized_l2_ranking': lmdi_values_normalized_l2_ranking,
         'baseline': lmdi_values_baseline,
         'mdi': mdi_values}
    lfi_rankings = \
        {'shap': shap_rankings,
         'signed_normalized_l2_avg': lmdi_rankings_signed_normalized_l2_avg,
         'signed_normalized_l2_noavg': lmdi_rankings_signed_normalized_l2_noavg,
         'signed_nonnormalized_l2_avg': lmdi_rankings_signed_nonnormalized_l2_avg,
         'signed_nonnormalized_l2_noavg':
             lmdi_rankings_signed_nonnormalized_l2_noavg,
         'nonl2_avg': lmdi_rankings_nonl2_avg,
         'nonl2_noavg': lmdi_rankings_nonl2_noavg,
         'l2_ranking': lmdi_rankings_l2_ranking,
         'nonl2_ranking': lmdi_rankings_nonl2_ranking,
         'normalized_l2_ranking': lmdi_values_normalized_l2_ranking,
         'baseline': lmdi_rankings_baseline,
         'mdi': mdi_rankings}
    
    # end time
    end = time.time()
    
    print(f"Progress Message 4/5: LMDI+ values/rankings obtained.")
    print(f"Step #4 took {end-start} seconds.")
    
    result_dir = oj(os.path.dirname(os.path.realpath(__file__)),
                    f'results/{TASK}/test{int(use_test)}/n{n}/seed{seed}')
    
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