# import required packages
from imodels import get_clean_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV, LogisticRegression
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusClassifier, RandomForestPlusRegressor
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import  AloRFPlusMDI, RFPlusMDI
import shap
from subgroup_detection import *
import warnings
import argparse
import os
from os.path import join as oj
warnings.filterwarnings('ignore', category=DeprecationWarning)

def split_data(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                        test_size=0.25,
                                                        random_state=seed)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def evaluate_model(X_train, X_valid, X_test, y_train, y_valid, y_test, task):
    
    # fit RF model
    if task == 'regression':
        rf = RandomForestRegressor(n_estimators=100, random_state=0)
    else:
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        
    rf.fit(X_train, y_train)
    
    print("Initial Random Forest fit for", task, "task.")
    
    # check performance on test set
    y_pred = rf.predict(X_test)

    # compute accuracy on the test set
    if task == 'regression':
        global_error_rf = np.sum((y_pred - y_test)**2)
    else:
        global_error_rf = np.sum(y_pred != y_test)
    
    # get feature importances
    explainer = shap.TreeExplainer(rf)
    if task == 'regression':
        shap_values = np.abs(explainer.shap_values(X_train, check_additivity=False))
    else:
        shap_values = np.abs(explainer.shap_values(X_train, check_additivity=False))[:,:,0]
    shap_rankings = np.argsort(-shap_values, axis = 1)
    
    # get rbo distance matrix
    shap_rbo_train = compute_rbo_matrix(shap_rankings, form = 'distance')
    shap_copy = pd.DataFrame(shap_values, columns=X_train.columns).copy()
    if task == "regression":
        shap_valid_values = np.abs(explainer.shap_values(X_valid,
                                                        check_additivity=False))
    else:
        shap_valid_values = np.abs(explainer.shap_values(X_valid,
                                                    check_additivity=False))[:,:,0]
    shap_valid_rankings = np.argsort(-shap_valid_values, axis = 1)
    best_error = np.inf
    opt_num_clusters = -1
    time_since_king = 0
    if task == 'regression':
        print("Total Squared Error for Zero Clusters:", np.sum((rf.predict(X_valid) - y_valid)**2))
    else:
        print("Total Error (# Misclassified) for Zero Clusters:", np.sum(rf.predict(X_valid) != y_valid))
    
    no_valid = False
    for num_clusters in np.arange(2, X_train.shape[0]//30):
        print("Now Calculating for", num_clusters, "Clusters")
        shap_train_clusters = assign_training_clusters(shap_rbo_train, num_clusters)
        valid_clusters = assign_testing_clusters(method="centroid",
                                                median_approx=True,
                                                rbo_distance_matrix=shap_rbo_train,
                                                lfi_train_ranking=shap_rankings,
                                                lfi_test_ranking=shap_valid_rankings,
                                                clusters = shap_train_clusters)
        total_error = 0
        for cluster in np.arange(1, num_clusters + 1):
            
            # if the cluster has zero validation points, break out of both loops
            if np.sum(valid_clusters == cluster) == 0:
                no_valid = True
                break
            
            if task == 'regression':
                local_rf = RandomForestRegressor(n_estimators=100, random_state=0)
            else:
                local_rf = RandomForestClassifier(n_estimators=100, random_state=0)
            local_rf.fit(X_train[shap_train_clusters == cluster], y_train[shap_train_clusters == cluster])
            local_preds = local_rf.predict(X_valid[valid_clusters == cluster])
            
            if task == 'regression':
                local_error = np.sum((local_preds - y_valid[valid_clusters == cluster])**2)
            else:
                local_error = np.sum(local_preds != y_valid[valid_clusters == cluster])
                
            total_error += local_error
            
        if no_valid:
            break
            
        if total_error < best_error:
            best_error = total_error
            opt_num_clusters = num_clusters
            time_since_king = 0
        else:
            time_since_king += 1
        if task == 'regression':
            print("Total Squared Error w/", num_clusters, "Clusters:", total_error)
        else:
            print("Total Error (# Misclassified) w/", num_clusters, "Clusters:", total_error)
        if time_since_king > 2:
            break
    print(f'Optimal Number of Clusters: {opt_num_clusters}')
    
    # fit rf+
    if task == 'regression':
        rf_plus = RandomForestPlusRegressor(rf, prediction_model = RidgeCV())
    else:
        rf_plus = RandomForestPlusClassifier(rf, prediction_model = LogisticRegression())
    rf_plus.fit(X_train, y_train)

    # check performance on test set
    y_pred_rf_plus = rf_plus.predict(X_test)

    # compute accuracy on the test set
    if task == 'regression':
        global_error_rf_plus = np.sum((y_pred_rf_plus - y_test)**2)
    else:
        global_error_rf_plus = np.sum(y_pred_rf_plus != y_test)
        
    # compute auroc on test set
    if task == 'classification':
        auroc_rf_plus = roc_auc_score(y_test, rf_plus.predict_proba(X_test)[:,1])
        auprc_rf_plus = average_precision_score(y_test, rf_plus.predict_proba(X_test)[:,1])
        f1_rf_plus = f1_score(y_test, rf_plus.predict(X_test))
        
    # get feature importances
    mdi_explainer = RFPlusMDI(rf_plus, evaluate_on='oob')
    mdi = np.abs(mdi_explainer.explain_linear_partial(np.asarray(X_train), y_train, l2norm = True))
    mdi_rankings = mdi_explainer.get_rankings(mdi)
    
    # get rbo distance matrix
    mdi_rbo_train = compute_rbo_matrix(mdi_rankings, form = 'distance')
    mdi_copy = pd.DataFrame(mdi, columns=X_train.columns).copy()
    mdi_train_clusters = assign_training_clusters(mdi_rbo_train, opt_num_clusters)
    
    # get mdi rankings assignments for test points
    mdi_test = np.abs(mdi_explainer.explain_linear_partial(np.asarray(X_test), l2norm=True))
    mdi_test_rankings = mdi_explainer.get_rankings(mdi_test)
    
    mdi_test_clusters = assign_testing_clusters(method = "centroid", median_approx = False,
                                     rbo_distance_matrix = mdi_rbo_train,
                                     lfi_train_ranking = mdi_rankings,
                                     lfi_test_ranking = mdi_test_rankings,
                                     clusters = mdi_train_clusters)
    
    total_local_error_rf_plus = 0
    if task == 'classification':
        local_rf_plus_aurocs = []
        local_rf_plus_auprcs = []
        local_rf_plus_f1s = []
    for cluster in np.arange(1, opt_num_clusters + 1):
        if task == 'classification':
            local_rf_plus = RandomForestPlusClassifier(RandomForestClassifier(n_estimators=100, random_state=0), prediction_model = LogisticRegression())
        else:
            local_rf_plus = RandomForestPlusRegressor(RandomForestRegressor(n_estimators=100, random_state=0), prediction_model = RidgeCV())
        local_rf_plus.fit(X_train[mdi_train_clusters == cluster], y_train[mdi_train_clusters == cluster])
        local_preds = local_rf_plus.predict(X_test[mdi_test_clusters == cluster])
        if task == 'regression':
            # if there are no test points in the cluster, the local error is zero
            if len(y_test[mdi_test_clusters == cluster]) == 0:
                local_error = 0
            else:
                local_error = np.sum((local_preds - y_test[mdi_test_clusters == cluster])**2)
        else:
            if len(y_test[mdi_test_clusters == cluster]) == 0:
                local_error = 0
            else:
                local_error = np.sum(local_preds != y_test[mdi_test_clusters == cluster])
            local_rf_plus_aurocs.append(roc_auc_score(y_test[mdi_test_clusters == cluster],
                                                        local_rf_plus.predict_proba(X_test[mdi_test_clusters == cluster])[:,1]))
            local_rf_plus_auprcs.append(average_precision_score(y_test[mdi_test_clusters == cluster],
                                                                local_rf_plus.predict_proba(X_test[mdi_test_clusters == cluster])[:,1]))
            local_rf_plus_f1s.append(f1_score(y_test[mdi_test_clusters == cluster],
                                            local_rf_plus.predict(X_test[mdi_test_clusters == cluster]))
                                     )
        total_local_error_rf_plus += local_error
        
    # get feature importances
    mdi_explainer_intercept = RFPlusMDI(rf_plus, evaluate_on='oob')
    mdi_intercept = np.abs(mdi_explainer_intercept.explain_linear_partial(np.asarray(X_train), y_train, l2norm = True))
    mdi_intercept_rankings = mdi_explainer_intercept.get_rankings(mdi_intercept)
    
    # get rbo distance matrix
    mdi_rbo_train_int = compute_rbo_matrix(mdi_intercept_rankings, form = 'distance')
    mdi_copy_int = pd.DataFrame(mdi_intercept, columns=X_train.columns).copy()
    mdi_train_clusters_int = assign_training_clusters(mdi_rbo_train_int, opt_num_clusters)
    
    # get mdi rankings assignments for test points
    mdi_test_int = np.abs(mdi_explainer_intercept.explain_linear_partial(np.asarray(X_test), l2norm=True))
    mdi_test_rankings_int = mdi_explainer_intercept.get_rankings(mdi_test_int)
    
    mdi_test_clusters_int = assign_testing_clusters(method = "centroid", median_approx = False,
                                     rbo_distance_matrix = mdi_rbo_train_int,
                                     lfi_train_ranking = mdi_intercept_rankings,
                                     lfi_test_ranking = mdi_test_rankings_int,
                                     clusters = mdi_train_clusters_int)
    
    total_local_int_error_rf_plus = 0
    if task == 'classification':
        local_rf_plus_aurocs = []
        local_rf_plus_auprcs = []
        local_rf_plus_f1s = []
    for cluster in np.arange(1, opt_num_clusters + 1):
        if task == 'classification':
            local_rf_plus = RandomForestPlusClassifier(RandomForestClassifier(n_estimators=100, random_state=0), prediction_model = LogisticRegression())
        else:
            local_rf_plus = RandomForestPlusRegressor(RandomForestRegressor(n_estimators=100, random_state=0), prediction_model = RidgeCV())
        local_rf_plus.fit(X_train[mdi_train_clusters == cluster], y_train[mdi_train_clusters == cluster])
        local_preds = local_rf_plus.predict(X_test[mdi_test_clusters_int == cluster])
        if task == 'regression':
            # if there are no test points in the cluster, the local error is zero
            if len(y_test[mdi_test_clusters_int == cluster]) == 0:
                local_error = 0
            else:
                local_error = np.sum((local_preds - y_test[mdi_test_clusters_int == cluster])**2)
        else:
            if len(y_test[mdi_test_clusters_int == cluster]) == 0:
                local_error = 0
            else:
                local_error = np.sum(local_preds != y_test[mdi_test_clusters_int == cluster])
            local_rf_plus_aurocs.append(roc_auc_score(y_test[mdi_test_clusters_int == cluster],
                                                        local_rf_plus.predict_proba(X_test[mdi_test_clusters_int == cluster])[:,1]))
            local_rf_plus_auprcs.append(average_precision_score(y_test[mdi_test_clusters_int == cluster],
                                                                local_rf_plus.predict_proba(X_test[mdi_test_clusters_int == cluster])[:,1]))
            local_rf_plus_f1s.append(f1_score(y_test[mdi_test_clusters_int == cluster],
                                            local_rf_plus.predict(X_test[mdi_test_clusters_int == cluster]))
                                     )
        total_local_int_error_rf_plus += local_error
    
    if task == 'classification':
        auroc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
        auprc_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:,1])
        f1_rf = f1_score(y_test, rf.predict(X_test))
        local_rf_aurocs = []
        local_rf_auprcs = []
        local_rf_f1s = []
    
    shap_copy = pd.DataFrame(shap_values, columns=X_train.columns).copy()
    shap_train_clusters = assign_training_clusters(shap_rbo_train, opt_num_clusters)
    if task == "regression":
        shap_test_values = np.abs(explainer.shap_values(X_test,
                                                        check_additivity=False))
    else:
        shap_test_values = np.abs(explainer.shap_values(X_test,
                                                    check_additivity=False))[:,:,0]
    shap_test_rankings = np.argsort(-shap_test_values, axis = 1)
    shap_test_clusters = assign_testing_clusters(method="centroid", median_approx=False,
                                            rbo_distance_matrix=shap_rbo_train,
                                            lfi_train_ranking=shap_rankings,
                                            lfi_test_ranking=shap_test_rankings,
                                            clusters = shap_train_clusters)
    
    total_local_error_rf = 0
    for cluster in np.arange(1, opt_num_clusters + 1):
        if task == "classification":
            local_rf = RandomForestClassifier(n_estimators=100, random_state=0)
        else:
            local_rf = RandomForestRegressor(n_estimators=100, random_state=0)
        local_rf.fit(X_train[shap_train_clusters == cluster], y_train[shap_train_clusters == cluster])
        local_preds = local_rf.predict(X_test[shap_test_clusters == cluster])
        if task == 'regression':
            if len(y_test[shap_test_clusters == cluster]) == 0:
                local_error = 0
            else:
                local_error = np.sum((local_preds - y_test[shap_test_clusters == cluster])**2)
        else:
            if len(y_test[shap_test_clusters == cluster]) == 0:
                local_error = 0
            else:
                local_error = np.sum(local_preds != y_test[shap_test_clusters == cluster])
            local_rf_aurocs.append(roc_auc_score(y_test[shap_test_clusters == cluster],
                                                local_rf.predict_proba(X_test[shap_test_clusters == cluster])[:,1]))
            local_rf_auprcs.append(average_precision_score(y_test[shap_test_clusters == cluster],
                                                        local_rf.predict_proba(X_test[shap_test_clusters == cluster])[:,1]))
            local_rf_f1s.append(f1_score(y_test[shap_test_clusters == cluster],
                                        local_rf.predict(X_test[shap_test_clusters == cluster]))
                                )
        total_local_error_rf += local_error
        
    if task == 'classification':
        rf_plus_weighted_auroc = weighted_metric(local_rf_plus_aurocs, np.bincount(mdi_test_clusters.astype(int))[1:])
        rf_plus_weighted_auprc = weighted_metric(local_rf_plus_auprcs, np.bincount(mdi_test_clusters.astype(int))[1:])
        rf_plus_weighted_f1 = weighted_metric(local_rf_plus_f1s, np.bincount(mdi_test_clusters.astype(int))[1:])
        rf_weighted_auroc = weighted_metric(local_rf_aurocs, np.bincount(shap_test_clusters.astype(int))[1:])
        rf_weighted_auprc = weighted_metric(local_rf_auprcs, np.bincount(shap_test_clusters.astype(int))[1:])
        rf_weighted_f1 = weighted_metric(local_rf_f1s, np.bincount(shap_test_clusters.astype(int))[1:])
        return {'global_error_rf': global_error_rf,
                'global_error_rf_plus': global_error_rf_plus,
                'total_local_error_rf': total_local_error_rf,
                'total_local_error_rf_plus': total_local_error_rf_plus,
                'total_local_int_error_rf_plus': total_local_int_error_rf_plus,
                'auroc_rf': auroc_rf,
                'auprc_rf': auprc_rf,
                'f1_rf': f1_rf,
                'auroc_rf_plus': auroc_rf_plus,
                'auprc_rf_plus': auprc_rf_plus,
                'f1_rf_plus': f1_rf_plus,
                'rf_plus_weighted_auroc': rf_plus_weighted_auroc,
                'rf_plus_weighted_auprc': rf_plus_weighted_auprc,
                'rf_plus_weighted_f1': rf_plus_weighted_f1,
                'rf_weighted_auroc': rf_weighted_auroc,
                'rf_weighted_auprc': rf_weighted_auprc,
                'rf_weighted_f1': rf_weighted_f1}
    
    return {'global_error_rf': global_error_rf,
            'global_error_rf_plus': global_error_rf_plus,
            'total_local_error_rf': total_local_error_rf,
            'total_local_error_rf_plus': total_local_error_rf_plus,
            'total_local_int_error_rf_plus': total_local_int_error_rf_plus}
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_dir = os.getenv("SCRATCH")
    if default_dir is not None:
        default_dir = oj(default_dir, "feature_importance", "subgroup", "results")
    else:
        default_dir = oj(os.path.dirname(os.path.realpath(__file__)), 'results')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--datasource', type=str, default=None)
    parser.add_argument('--dataname', type=str, default=None)
    args = parser.parse_args()
    
    # Convert Namespace to a dictionary
    args_dict = vars(args)

    # Assign each key-value pair to a variable
    seed = args_dict['seed']
    datasource = args_dict['datasource']
    dataname = args_dict['dataname']
    print("Obtaining", dataname, "from", datasource, "with seed", seed)
    if datasource == "openml":
        dataname = int(dataname)
    X, y, feature_names = get_clean_dataset(dataname, data_source = datasource)
    X = pd.DataFrame(X, columns=feature_names)
    print("Data obtained!")
    
    # check if task is regression or classification
    if len(np.unique(y)) == 2:
        task = 'classification'
    else:
        task = 'regression'
        # convert y to float
        y = y.astype(float)
    
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y, seed)
    print("data split")
    eval_dict = evaluate_model(X_train, X_valid, X_test, y_train, y_valid, y_test, task)
    eval_dict['seed'] = seed
    eval_dict['datasource'] = datasource
    eval_dict['dataname'] = dataname
    eval_dict['task'] = task
    # convert eval_dict to dataframe and save dataframe to csv
    eval_df = pd.DataFrame([eval_dict])
    print(default_dir)
    eval_df.to_csv(oj(default_dir, f'new_{datasource}_{dataname}_{seed}.csv'), index=False)
    print("ran experiment successfully")