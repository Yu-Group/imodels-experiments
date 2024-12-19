# import required packages
from imodels import get_clean_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusRegressor, RandomForestPlusClassifier
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import  AloRFPlusMDI, RFPlusMDI
import shap
from subgroup_detection import *
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LogisticRegression

global_task = None

def split_data(X, y, seed = 1):
    # split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                        test_size=0.25,
                                                        random_state=seed)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def fit_models(X_train, y_train, task):
    # fit models
    if task == 'classification':
        global_task = 'classification'
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)
        rf_plus = RandomForestPlusClassifier(rf_model=rf,
                                        prediction_model=LogisticRegression())
        rf_plus.fit(X_train, y_train)
    elif task == 'regression':
        global_task = 'regression'
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train, y_train)
        rf_plus = RandomForestPlusRegressor(rf_model=rf,
                                            prediction_model=RidgeCV())
        rf_plus.fit(X_train, y_train)
    return rf, rf_plus

def get_shap(X, shap_explainer):
    if global_task == 'classification':
        shap_values = shap_explainer.shap_values(X, check_additivity=False)[:,:,1]
    else:
        shap_values = shap_explainer.shap_values(X, check_additivity=False)
    shap_rankings = np.argsort(-np.abs(shap_values), axis = 1)
    return shap_values, shap_rankings

def get_lmdi(X, y, lmdi_explainer):
    # get feature importances
    lmdi = np.abs(lmdi_explainer.explain_linear_partial(np.asarray(X), y,
                                                        l2norm=True))
    mdi_rankings = lmdi_explainer.get_rankings(lmdi)
    return lmdi, mdi_rankings

def get_num_clusters(X_train, y_train, X_valid, y_valid, shap_explainer,
                     shap_rbo_train, shap_train_rankings):
    if global_task == 'classification':
        shap_valid_values = np.abs(shap_explainer.shap_values(X_valid,
                                                        check_additivity=False))[:,:,1]
    else:
        shap_valid_values = np.abs(shap_explainer.shap_values(X_valid,
                                                    check_additivity=False))
    shap_valid_rankings = np.argsort(-shap_valid_values, axis = 1)
    
    
    lowest_error = np.inf
    opt_num_clusters = -1
    error_lst = []
    time_since_king = 0
    opt_clusters = None
    for num_clusters in np.arange(2, 21):
        shap_train_clusters = assign_training_clusters(shap_rbo_train, num_clusters)
        valid_clusters = assign_testing_clusters(method="centroid",
                                                median_approx=True,
                                                rbo_distance_matrix=shap_rbo_train,
                                                lfi_train_ranking=shap_train_rankings,
                                                lfi_test_ranking=shap_valid_rankings,
                                                clusters = shap_train_clusters)
        total_error = 0
        for cluster in np.arange(1, num_clusters + 1):
            if global_task == 'classification':
                local_rf = RandomForestClassifier(n_estimators=100, random_state=0)
            else:
                local_rf = RandomForestRegressor(n_estimators=100, random_state=0)
            local_rf.fit(X_train[shap_train_clusters == cluster], y_train[shap_train_clusters == cluster])
            local_preds = local_rf.predict(X_valid[valid_clusters == cluster])
            if global_task == 'classification':
                local_error = np.sum(local_preds != y_valid[valid_clusters == cluster])
            else:
                local_error = np.sum((local_preds - y_valid[valid_clusters == cluster])**2)
            total_error += local_error
        if total_error < lowest_error:
            lowest_error = total_error
            opt_num_clusters = num_clusters
            time_since_king = 0
            opt_clusters = shap_train_clusters
        else:
            time_since_king += 1
        if time_since_king > 2:
            break
    return opt_num_clusters, opt_clusters
        
def run_experiment(X, y, task, seed = 1):
    # split data
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y, seed)
    # fit models
    rf, rf_plus = fit_models(X_train, y_train, task)
    # get shap values
    shap_explainer = shap.TreeExplainer(rf)
    shap_values, shap_rankings = get_shap(X_train, shap_explainer)
    # get lmdi values
    lmdi_explainer = RFPlusMDI(rf_plus)
    lmdi_train, lmdi_train_rankings = get_lmdi(X_train, y_train, lmdi_explainer)
    # get shap rbo
    shap_rbo_train = compute_rbo_matrix(shap_rankings)
    # get optimal number of clusters
    opt_num_clusters, opt_clusters = get_num_clusters(X_train, y_train, X_valid,
                                                      y_valid, shap_explainer,
                                                      shap_rbo_train, shap_rankings)
    lmdi_rbo_train = compute_rbo_matrix(lmdi_train_rankings)
    lmdi_train_clusters = assign_training_clusters(lmdi_rbo_train, opt_num_clusters)
    lmdi_test, lmdi_test_rankings = get_lmdi(X_test, y_test, lmdi_explainer)
    lmdi_test_clusters = assign_testing_clusters(method="centroid",
                                                 median_approx=False,
                                                 rbo_distance_matrix=lmdi_rbo_train,
                                                    lfi_train_ranking=lmdi_train_rankings,
                                                    lfi_test_ranking=lmdi_test_rankings,
                                                    clusters=lmdi_train_clusters)
    
    return opt_num_clusters, opt_clusters