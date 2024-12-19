# imports
import numpy as np
import pandas as pd
from subgroup_detection import *
from subgroup_experiment import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from imodels.tree.rf_plus.rf_plus.rf_plus_models import \
    RandomForestPlusRegressor, RandomForestPlusClassifier
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import \
    RFPlusMDI, AloRFPlusMDI
from scipy import cluster
from scipy.cluster.hierarchy import fcluster, cut_tree
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, \
    accuracy_score, r2_score, f1_score, log_loss, root_mean_squared_error
from imodels import get_clean_dataset
import openml
from ucimlrepo import fetch_ucirepo
import argparse
import os
from os.path import join as oj

def get_openml_data(id, num_samples=2000):
    
    # check that the dataset_id is in the set of tested datasets
    known_ids = {361247, 361243, 361242, 361251, 361253, 361260, 361259, 361256,
                 361254, 361622}
    if id not in known_ids:
        raise ValueError(f"Data ID {id} is not in the set of known datasets.")
    
    # get the dataset from openml
    task = openml.tasks.get_task(id)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    # subsample the data if necessary
    if num_samples is not None and num_samples < X.shape[0]:
        X = X.sample(num_samples)
        y = y.loc[X.index]
    
    # reset the index of X and y
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # convert X and y to numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()
    
    # perform transformations if needed
    log_transform = {361260, 361622}
    if id in log_transform:
        y = np.log(y)
    
    return X, y

def fit_models(X_train, y_train, task):
    # fit models
    if task == "classification":
        rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3,
                                    max_features='sqrt', random_state=42)
        rf.fit(X_train, y_train)
        rf_plus_baseline = RandomForestPlusClassifier(rf_model=rf,
                                        include_raw=False, fit_on="inbag",
                                        prediction_model=LogisticRegression())
        rf_plus_baseline.fit(X_train, y_train)
        rf_plus = RandomForestPlusClassifier(rf_model=rf)
        rf_plus.fit(X_train, y_train)
    elif task == "regression":
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                   max_features=0.33, random_state=42)
        rf.fit(X_train, y_train)
        rf_plus_baseline = RandomForestPlusRegressor(rf_model=rf,
                                        include_raw=False, fit_on="inbag",
                                        prediction_model=LinearRegression())
        rf_plus_baseline.fit(X_train, y_train)
        rf_plus = RandomForestPlusRegressor(rf_model=rf)
        rf_plus.fit(X_train, y_train)
    else:
        raise ValueError("Task must be 'classification' or 'regression'.")
    return rf, rf_plus_baseline, rf_plus

def get_shap(X, shap_explainer, task):
    if task == "classification":
        # the shap values are an array of shape
        # (# of samples, # of features, # of classes), and in this binary
        # classification case, we want the shap values for the positive class.
        # check_additivity=False is used to speed up computation.
        shap_values = \
            shap_explainer.shap_values(X, check_additivity = False)[:, :, 1]
    elif task == "regression":
        # check_additivity=False is used to speed up computation.
        shap_values = shap_explainer.shap_values(X, check_additivity = False)
    else:
        raise ValueError("Task must be 'classification' or 'regression'.")
    # get the rankings of the shap values. negative absolute value is taken
    # because np.argsort sorts from smallest to largest.
    shap_rankings = np.argsort(-np.abs(shap_values), axis = 1)
    return shap_values, shap_rankings

def get_lmdi_explainers(rf_plus, lmdi_variants, rf_plus_baseline = None):
    # create the lmdi explainer objects
    lmdi_explainers = {}
    if rf_plus_baseline is not None:
        lmdi_explainers["baseline"] = RFPlusMDI(rf_plus_baseline,
                                                mode = "only_k",
                                                evaluate_on = "inbag")
    for variant_name in lmdi_variants.keys():
        if lmdi_variants[variant_name]["loo"]:
            lmdi_explainers[variant_name] = AloRFPlusMDI(rf_plus,
                                                         mode = "only_k")
        else:
            lmdi_explainers[variant_name] = RFPlusMDI(rf_plus, mode = "only_k")
    return lmdi_explainers
    

def get_lmdi(X, y, lmdi_variants, lmdi_explainers):
    
    # initialize storage mappings
    lmdi_values = {}
    lmdi_rankings = {}
    
    # check if the explainer mapping has a "baseline"
    if len(lmdi_explainers) == len(lmdi_variants) + 1 and \
        "baseline" in lmdi_explainers:
        lmdi_values["baseline"] = lmdi_explainers["baseline"].explain_linear_partial(X, y,
                                            l2norm=False, sign=False,
                                            normalize=False,leaf_average=False)
        lmdi_rankings["baseline"] = lmdi_explainers["baseline"].get_rankings(np.abs(lmdi_values["baseline"]))
    
    for name, explainer in lmdi_explainers.items():
        if name == "baseline":
            continue
        variant_args = lmdi_variants[name]        
        lmdi_values[name] = explainer.explain_linear_partial(X, y,
                                            l2norm=variant_args["l2norm"],
                                            sign=variant_args["sign"],
                                            normalize=variant_args["normalize"],
                                            leaf_average=variant_args["leaf_average"])
        lmdi_rankings[name] = explainer.get_rankings(np.abs(lmdi_values[name]))
        
    return lmdi_values, lmdi_rankings

def get_train_clusters(lfi_train_values, method):
    # make sure method is valid
    if method not in ["kmeans", "hierarchical"]:
        raise ValueError("Method must be 'kmeans' or 'hierarchical'.")
    if method == "hierarchical":
        train_linkage = {}
        for method, values in lfi_train_values.items():
            train_linkage[method] = cluster.hierarchy.ward(values)
        train_clusters = {}
        for method, link in train_linkage.items():
            num_cluster_map = {}
            for num_clusters in range(2, 11):
                num_cluster_map[num_clusters] = cut_tree(link, n_clusters=num_clusters).flatten()
            train_clusters[method] = num_cluster_map
    elif method == "kmeans":
        train_clusters = {}
        for method, values in lfi_train_values.items():
            num_cluster_map = {}
            for num_clusters in range(2, 11):
                centroids, _ = cluster.vq.kmeans(obs=values, k_or_guess=num_clusters)
                kmeans, _ = cluster.vq.vq(values, centroids)
                num_cluster_map[num_clusters] = kmeans
            train_clusters[method] = num_cluster_map
    train_clusters_final = {}
    for method, clusters in train_clusters.items():
        num_cluster_map = {}
        for num_clusters, cluster_labels in clusters.items():
            cluster_map = {}
            for cluster_num in range(num_clusters):
                cluster_indices = np.where(cluster_labels == cluster_num)[0]
                cluster_map[cluster_num] = cluster_indices
            num_cluster_map[num_clusters] = cluster_map
        train_clusters_final[method] = num_cluster_map
    return train_clusters, train_clusters_final

def get_cluster_centroids(lfi_train_values, train_clusters):
    # for each method, for each number of clusters, get the clusters and compute their centroids
    cluster_centroids = {}
    for method, clusters in train_clusters.items():
        num_cluster_centroids = {}
        for num_clusters, cluster_labels in clusters.items():
            centroids = np.zeros((num_clusters, X.shape[1]))
            for cluster_num in range(num_clusters):
                cluster_indices = np.where(cluster_labels == cluster_num)[0]
                cluster_values = lfi_train_values[method][cluster_indices]
                centroids[cluster_num] = np.mean(cluster_values, axis = 0)
            num_cluster_centroids[num_clusters] = centroids
        cluster_centroids[method] = num_cluster_centroids
    return cluster_centroids

def get_test_clusters(lfi_test_values, cluster_centroids):
    # for each method, for its test values, assign the test values to the closest centroid
    test_value_clusters = {}
    for method, centroids in cluster_centroids.items():
        num_cluster_map = {}
        for num_clusters, centroid_values in centroids.items():
            cluster_membership = np.zeros(len(lfi_test_values[method]))
            for i, test_value in enumerate(lfi_test_values[method]):
                distances = np.linalg.norm(centroid_values - test_value, axis=1)
                cluster_membership[i] = np.argmin(distances)
            num_cluster_map[num_clusters] = cluster_membership
        test_value_clusters[method] = num_cluster_map
    test_clusters = {}
    for method, clusters in test_value_clusters.items():
        num_cluster_map = {}
        for num_clusters, cluster_labels in clusters.items():
            cluster_map = {}
            for cluster_num in range(num_clusters):
                cluster_indices = np.where(cluster_labels == cluster_num)[0]
                cluster_map[cluster_num] = cluster_indices
            num_cluster_map[num_clusters] = cluster_map
        test_clusters[method] = num_cluster_map
    return test_clusters
    

if __name__ == '__main__':
    
    # store command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dataid', type=int, default=None)
    parser.add_argument('--clustertype', type=str, default=None)
    parser.add_argument('--njobs', type=int, default=1)
    args = parser.parse_args()
    
    # convert namespace to a dictionary
    args_dict = vars(args)

    # assign the arguments to variables
    seed = args_dict['seed']
    dataid = args_dict['dataid']
    clustertype = args_dict['clustertype']
    njobs = args_dict['njobs']
    
    # get data
    X, y = get_openml_data(dataid)
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                        random_state=seed)
    
    # check if task is regression or classification
    if len(np.unique(y)) == 2:
        task = 'classification'
    else:
        task = 'regression'
        
    # fit the prediction models
    rf, rf_plus_baseline, rf_plus = fit_models(X_train, y_train, task)
    
    # obtain shap feature importances
    shap_explainer = shap.TreeExplainer(rf)
    shap_train_values, shap_train_rankings = get_shap(X_train, shap_explainer,
                                                      task)
    shap_test_values, shap_test_rankings = get_shap(X_test, shap_explainer,
                                                    task)
    
    # create list of lmdi variants
    loo = {True: "aloo", False: "nonloo"}
    l2norm = {True: "l2", False: "nonl2"}
    sign = {True: "signed", False: "unsigned"}
    normalize = {True: "normed", False: "nonnormed"}
    leaf_average = {True: "leafavg", False: "noleafavg"}
    ranking = {True: "rank", False: "norank"}
    lmdi_variants = {}
    for l in loo:
        for n in l2norm:
            for s in sign:
                for nn in normalize:
                    # sign and normalize are only relevant if l2norm is True
                    if (not n) and (s or nn):
                        continue
                    for la in leaf_average:
                        for r in ranking:
                            variant_name = f"{loo[l]}_{l2norm[n]}_{sign[s]}_{normalize[nn]}_{leaf_average[la]}_{ranking[r]}"
                            arg_map = {"loo": l, "l2norm": n, "sign": s,
                                       "normalize": nn, "leaf_average": la,
                                       "ranking": r}
                            lmdi_variants[variant_name] = arg_map
    
    # obtain lmdi feature importances
    lmdi_explainers = get_lmdi_explainers(rf_plus, lmdi_variants,
                                          rf_plus_baseline = rf_plus_baseline)
    lfi_train_values, lfi_train_rankings = get_lmdi(X_train, y_train,
                                                    lmdi_variants,
                                                    lmdi_explainers)
    lfi_test_values, lfi_test_rankings = get_lmdi(X_test, None,
                                                  lmdi_variants,
                                                  lmdi_explainers)
    # add shap to the dictionaries
    lfi_train_values["shap"] = shap_train_values
    lfi_train_rankings["shap"] = shap_train_rankings
    lfi_test_values["shap"] = shap_test_values
    lfi_test_rankings["shap"] = shap_test_rankings
    
    # get the clusterings
    train_clusters_for_centroids, train_clusters = get_train_clusters(lfi_train_values, clustertype)
    cluster_centroids = get_cluster_centroids(lfi_train_values, train_clusters_for_centroids)
    test_clusters = get_test_clusters(lfi_test_values, cluster_centroids)
    
    # create a mapping of metrics to measure
    if task == "classification":
        metrics = {"accuracy": accuracy_score, "roc_auc": roc_auc_score,
                   "average_precision": average_precision_score,
                   "f1": f1_score, "log_loss": log_loss}
    else:
        metrics = {"r2": r2_score, "rmse": root_mean_squared_error}
    
    # for each method, for each number of clusters,
    # train a linear model on the training set for each cluster and
    # use it to predict the testing set for each cluster. save the results.
    metrics_to_methods = {}
    for metric_name, metric_func in metrics.items():
        metrics_to_methods[metric_name] = {}
        for method in train_clusters.keys():
            methods_to_scores = {}
            for num_clusters in range(2, 11):
                cluster_scores = []
                cluster_sizes = []
                for cluster_idx in range(num_clusters):
                    X_cluster_train = X_train[train_clusters[method][num_clusters][cluster_idx]]
                    y_cluster_train = y_train[train_clusters[method][num_clusters][cluster_idx]]
                    X_cluster_test = X_test[test_clusters[method][num_clusters][cluster_idx]]
                    y_cluster_test = y_test[test_clusters[method][num_clusters][cluster_idx]]
                    if X_cluster_test.shape[0] == 0:
                        continue
                    if task == "classification":
                        model = LogisticRegression()
                    else:
                        model = LinearRegression()
                    model.fit(X_cluster_train, y_cluster_train)
                    # print("Method:", method, "; # Clusters:", num_clusters, "; Cluster:", cluster_idx)
                    # print(X_cluster_test.shape)
                    # print(X_cluster_train.shape)
                    y_cluster_pred = model.predict(X_cluster_test)
                    cluster_scores.append(metric_func(y_cluster_test, y_cluster_pred))
                    cluster_sizes.append(X_cluster_test.shape[0])
                methods_to_scores[num_clusters] = \
                    weighted_metric(np.array(cluster_scores), np.array(cluster_sizes))
            # average accuracy across clusters
            metrics_to_methods[metric_name][method] = methods_to_scores
    
    # save the results
    result_dir = oj(os.path.dirname(os.path.realpath(__file__)), 'results/')
    # print(result_dir)
    # print(metrics_to_methods)
    for metric_name in metrics_to_methods.keys():
        # write the results to a csv file
        print(f"Saving {metric_name} results...")
        for method in metrics_to_methods[metric_name].keys():
            print("Method:", method)
            # print(metrics_to_methods[metric_name])
            df = pd.DataFrame(list(metrics_to_methods[metric_name][method].items()), columns=["nclust", f"{metric_name}"])
            # print(df)
            if not os.path.exists(oj(result_dir, f"dataid{dataid}/seed{seed}/metric{metric_name}/{clustertype}")):
                os.makedirs(oj(result_dir, f"dataid{dataid}/seed{seed}/metric{metric_name}/{clustertype}"))
            df.to_csv(oj(result_dir,
                    f"dataid{dataid}/seed{seed}/metric{metric_name}/{clustertype}", f"{method}.csv"))
    
    print("Results saved!")