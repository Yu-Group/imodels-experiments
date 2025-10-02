# data science imports
import numpy as np
import pandas as pd

# for saving results
import argparse
import os
from os.path import join as oj

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
import shap
import lime
import time

from sklearn.linear_model import LinearRegression, ElasticNetCV, LogisticRegression, LogisticRegressionCV

from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusClassifier, RandomForestPlusRegressor
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import LMDIPlus
from local_mdi import local_mdi_score

def read_data(data_id):
    """
    Reads in the X and y data corresponding to the data_id from the data/ dir.
    """
    X = np.loadtxt(oj("data", f"{data_id}/X.csv"), delimiter=",")
    y = np.loadtxt(oj("data", f"{data_id}/y.csv"), delimiter=",")
    
    # sample 1000 rows of X and y if X has more than 1000 rows
    if X.shape[0] > 1000:
        np.random.seed(42)
        indices = np.random.choice(X.shape[0], 1000, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y
    
def fit_rf_model(X, y, is_classification, n_estimators, min_samples_leaf, max_features):
    """
    Fits a Random Forest model to the data.
    """
    if is_classification:
        rf_model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                          max_features=max_features, random_state=42)
    else:
        rf_model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                         max_features=max_features, random_state=42)

    rf_model.fit(X, y)
    return rf_model

def fit_gb_model(X, y, is_classification, n_estimators, min_samples_leaf, max_features):
    """
    Fits a Gradient Boosting model to the data.
    """
    if is_classification:
        gb_model = GradientBoostingClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                              max_features=max_features, random_state=42)
    else:
        gb_model = GradientBoostingRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                             max_features=max_features, random_state=42)

    gb_model.fit(X, y)
    return gb_model

def fit_rf_plus_elasticnet_model(X, y, rf_model, is_classification):
    if is_classification:
        rf_plus_model = RandomForestPlusClassifier(rf_model=rf_model,
                prediction_model=LogisticRegressionCV(penalty='elasticnet',
                    l1_ratios=[0.1,0.5,0.99], solver='saga', cv=3,
                    n_jobs=-1, tol=5e-4, max_iter=2000, random_state=42))
    else:
        rf_plus_model = RandomForestPlusRegressor(rf_model=rf_model,
                                    prediction_model=ElasticNetCV(cv=3,
                                    l1_ratio=[0.1,0.5,0.99],
                                    max_iter=2000,random_state=42))
    rf_plus_model.fit(X, y)
    
    return rf_plus_model

def fit_gb_plus_elasticnet_model(X, y, gb_model):
    
    # elastic net gb+
    gb_plus_model = RandomForestPlusRegressor(rf_model=gb_model,
                                    prediction_model=ElasticNetCV(cv=3,
                                    l1_ratio=[0.1,0.5,0.99],
                                    max_iter=2000,random_state=42))
    gb_plus_model.fit(X, y)

    return gb_plus_model

def get_shap(X, shap_explainer, is_classification):
    
    # check_additivity=False is used to speed up computation.
    if is_classification:
        shap_values = shap_explainer.shap_values(X, check_additivity=False)[:,:,1]
    else:
        shap_values = shap_explainer.shap_values(X)
    return shap_values

def get_lime(X: np.ndarray, rf_model, is_classification):
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
    mode = "classification" if is_classification else "regression"
    explainer = lime.lime_tabular.LimeTabularExplainer(X, verbose = False,
                                                       mode = mode)
    num_features = X.shape[1]
    for i in range(X.shape[0]):
        if mode == "classification":
            exp = explainer.explain_instance(X[i, :], rf_model.predict_proba,
                                         num_features = num_features)
        else:
            exp = explainer.explain_instance(X[i, :], rf_model.predict,
                                         num_features = num_features)
        original_feature_importance = exp.as_map()[1]
        sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
        for j in range(num_features):
            lime_values[i, j] = sorted_feature_importance[j][1] 
        
    return lime_values

def get_lmdi(X, lmdi_plus_explainer):
    
    # get feature importances
    lmdi_plus = lmdi_plus_explainer.get_lmdi_plus_scores(X)
        
    return lmdi_plus


if __name__ == "__main__":
    
    # store command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataid', type=int, default=None)
    parser.add_argument('--classification', type=int, default=None)
    parser.add_argument('--n_estimators', type=int, default=None)
    parser.add_argument('--min_samples_leaf', type=int, default=None)
    parser.add_argument('--max_features', type=str, default=None)
    args = parser.parse_args()
    
    # convert namespace to a dictionary
    args_dict = vars(args)

    # assign the arguments to variables
    data_id = args_dict['dataid']
    is_classification = args_dict['classification']
    n_estimators = args_dict['n_estimators']
    min_samples_leaf = args_dict['min_samples_leaf']
    max_features = args_dict['max_features']
    # convert to bool
    is_classification = bool(is_classification)
    print(f"Running runtime analysis for data_id {data_id} with classification={is_classification}")
    # convert max_features to float if not 'sqrt'
    if max_features != "sqrt":
        max_features = float(max_features)

    X, y = read_data(data_id)
    
    rf_start_time = time.time()
    
    rf_model = fit_rf_model(X, y, is_classification, n_estimators, min_samples_leaf, max_features)
    
    rf_end_time = time.time()
    
    rf_fitting_time = rf_end_time - rf_start_time
    
    print(f"Random Forest fitting time: {rf_fitting_time:.2f} seconds")

    # create elasticnet rf+ model
    
    rf_plus_start_time = time.time()
    
    rf_plus_model = fit_rf_plus_elasticnet_model(X, y, rf_model, is_classification)
    
    rf_plus_end_time = time.time()
    
    rf_plus_fitting_time = rf_plus_end_time - rf_plus_start_time
    
    print(f"Random Forest Plus fitting time: {rf_plus_fitting_time:.2f} seconds")
    
    # create shap explainer
    shap_rf_explainer_start_time = time.time()
    shap_rf_explainer = shap.TreeExplainer(rf_model)
    shap_rf_explainer_end_time = time.time()
    shap_rf_explainer_time = shap_rf_explainer_end_time - shap_rf_explainer_start_time
    print(f"SHAP explainer creation time: {shap_rf_explainer_time:.2f} seconds")
    
    # get shap values
    shap_rf_values_start_time = time.time()
    shap_rf_values = get_shap(X, shap_rf_explainer, is_classification)
    shap_rf_values_end_time = time.time()
    shap_rf_values_time = shap_rf_values_end_time - shap_rf_values_start_time
    print(f"SHAP values computation time: {shap_rf_values_time:.2f} seconds")
    
    # get lime values
    lime_rf_start_time = time.time()
    lime_rf_values = get_lime(X, rf_model, is_classification)
    lime_rf_end_time = time.time()
    lime_rf_time = lime_rf_end_time - lime_rf_start_time
    print(f"LIME values computation time: {lime_rf_time:.2f} seconds")
    
    # get local mdi values
    local_mdi_start_time = time.time()
    local_mdi_values = local_mdi_score(X, model=rf_model)
    local_mdi_end_time = time.time()
    local_mdi_time = local_mdi_end_time - local_mdi_start_time
    print(f"Local MDI values computation time: {local_mdi_time:.2f} seconds")

    # get lmdi plus explainer
    lmdi_plus_rf_explainer_start_time = time.time()
    lmdi_plus_rf_explainer = LMDIPlus(rf_plus_model, evaluate_on = "all")
    lmdi_plus_rf_explainer_end_time = time.time()
    lmdi_plus_rf_explainer_time = lmdi_plus_rf_explainer_end_time - lmdi_plus_rf_explainer_start_time
    print(f"LMDI+ explainer creation time: {lmdi_plus_rf_explainer_time:.2f} seconds")
    
    # get lmdi plus values
    lmdi_plus_rf_start_time = time.time()
    lmdi_plus_rf_values = get_lmdi(X, lmdi_plus_rf_explainer)
    lmdi_plus_rf_end_time = time.time()
    lmdi_plus_rf_values_time = lmdi_plus_rf_end_time - lmdi_plus_rf_start_time
    print(f"LMDI+ values computation time: {lmdi_plus_rf_values_time:.2f} seconds")
    
    # save results to df
    results_dir = oj("results", "rf", f"{data_id}/n_estimators_{n_estimators}/min_samples_leaf_{min_samples_leaf}/max_features_{max_features}")
    os.makedirs(results_dir, exist_ok=True)
    # make df with data_id and each run time
    results_df = pd.DataFrame({
        "data_id": [data_id],
        "rf_fitting_time": [rf_fitting_time],
        "rf_plus_fitting_time": [rf_plus_fitting_time],
        "shap_rf_explainer_time": [shap_rf_explainer_time],
        "shap_rf_values_time": [shap_rf_values_time],
        "lime_rf_time": [lime_rf_time],
        "local_mdi_time": [local_mdi_time],
        "lmdi_plus_rf_explainer_time": [lmdi_plus_rf_explainer_time],
        "lmdi_plus_rf_values_time": [lmdi_plus_rf_values_time]
    })
    
    results_df.to_csv(oj(results_dir, "runtime_results.csv"), index=False)

    print(f"RF results saved to {oj(results_dir, 'runtime_results.csv')}")
    print("gb runtime analysis completed successfully.")
    
    gb_start_time = time.time()
    
    gb_model = fit_gb_model(X, y, is_classification, n_estimators, min_samples_leaf, max_features)
    
    gb_end_time = time.time()
    
    gb_fitting_time = gb_end_time - gb_start_time
    
    print(f"Gradient boosting fitting time: {gb_fitting_time:.2f} seconds")

    # create elasticnet gb+ model
    
    gb_plus_start_time = time.time()
    
    gb_plus_model = fit_gb_plus_elasticnet_model(X, y, gb_model)
    
    gb_plus_end_time = time.time()
    
    gb_plus_fitting_time = gb_plus_end_time - gb_plus_start_time
    
    print(f"GB Plus fitting time: {gb_plus_fitting_time:.2f} seconds")
    
    # create shap explainer
    shap_gb_explainer_start_time = time.time()
    shap_gb_explainer = shap.TreeExplainer(gb_model)
    shap_gb_explainer_end_time = time.time()
    shap_gb_explainer_time = shap_gb_explainer_end_time - shap_gb_explainer_start_time
    print(f"SHAP explainer creation time: {shap_gb_explainer_time:.2f} seconds")
    
    # get shap values
    shap_gb_values_start_time = time.time()
    shap_gb_values = get_shap(X, shap_gb_explainer, is_classification=False) # gb is always regression
    shap_gb_values_end_time = time.time()
    shap_gb_values_time = shap_gb_values_end_time - shap_gb_values_start_time
    print(f"SHAP values computation time: {shap_gb_values_time:.2f} seconds")
    
    # get lime values
    lime_gb_start_time = time.time()
    lime_gb_values = get_lime(X, gb_model, is_classification=False) # gb is always regression
    lime_gb_end_time = time.time()
    lime_gb_time = lime_gb_end_time - lime_gb_start_time
    print(f"LIME values computation time: {lime_gb_time:.2f} seconds")

    # get lmdi plus explainer
    lmdi_plus_gb_explainer_start_time = time.time()
    lmdi_plus_gb_explainer = LMDIPlus(gb_plus_model, evaluate_on = "all")
    lmdi_plus_gb_explainer_end_time = time.time()
    lmdi_plus_gb_explainer_time = lmdi_plus_gb_explainer_end_time - lmdi_plus_gb_explainer_start_time
    print(f"LMDI+ explainer creation time: {lmdi_plus_gb_explainer_time:.2f} seconds")
    
    # get lmdi plus values
    lmdi_plus_gb_start_time = time.time()
    lmdi_plus_gb_values = get_lmdi(X, lmdi_plus_gb_explainer)
    lmdi_plus_gb_end_time = time.time()
    lmdi_plus_gb_values_time = lmdi_plus_gb_end_time - lmdi_plus_gb_start_time
    print(f"LMDI+ values computation time: {lmdi_plus_gb_values_time:.2f} seconds")
    
    # save results to df
    results_dir = oj("results", "gb", f"{data_id}/n_estimators_{n_estimators}/min_samples_leaf_{min_samples_leaf}/max_features_{max_features}")
    os.makedirs(results_dir, exist_ok=True)
    # make df with data_id and each run time
    results_df = pd.DataFrame({
        "data_id": [data_id],
        "gb_fitting_time": [gb_fitting_time],
        "gb_plus_fitting_time": [gb_plus_fitting_time],
        "shap_gb_explainer_time": [shap_gb_explainer_time],
        "shap_gb_values_time": [shap_gb_values_time],
        "lime_gb_time": [lime_gb_time],
        "lmdi_plus_gb_explainer_time": [lmdi_plus_gb_explainer_time],
        "lmdi_plus_gb_values_time": [lmdi_plus_gb_values_time]
    })
    
    results_df.to_csv(oj(results_dir, "runtime_results.csv"), index=False)

    print(f"gb results saved to {oj(results_dir, 'runtime_results.csv')}")
    print("gb runtime analysis completed successfully.")