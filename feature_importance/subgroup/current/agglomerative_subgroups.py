# import required packages
from imodels import get_clean_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, \
    accuracy_score, r2_score, f1_score, log_loss, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusRegressor, RandomForestPlusClassifier
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import RFPlusMDI, AloRFPlusMDI
import shap
from subgroup_detection import *
import warnings
from sklearn.linear_model import RidgeCV, LogisticRegressionCV, LogisticRegression, LinearRegression
import argparse
import os
from os.path import join as oj
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster, cut_tree
from scipy import cluster
from scipy.spatial.distance import squareform
import time
from joblib import Parallel, delayed
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# global variable for classification/regression status
TASK = None

def preprocessing_data_X(X):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X.select_dtypes(include=["number"]).columns
    if X[numerical_cols].isnull().any().any():
        num_imputer = SimpleImputer(strategy="mean")
        X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    if len(categorical_cols) > 0 and X[categorical_cols].isnull().any().any():
        # Convert categorical columns to string to ensure consistent types
        X[categorical_cols] = X[categorical_cols].astype(str)
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_categorical = encoder.fit_transform(X[categorical_cols])
        X_categorical_df = pd.DataFrame(
            X_categorical,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=X.index
        )
        X = pd.concat([X[numerical_cols], X_categorical_df], axis=1)
    else:
        X = X[numerical_cols]
    X = X.to_numpy()
    if X.shape[0]>2000:
        X = X[:2000,:]
    return X

def preprocessing_data_y(y):
    if y.to_numpy().shape[1] > 1:
        y = y.iloc[:, 0].to_numpy().flatten()
    else:
        y = y.to_numpy().flatten()
    if y.shape[0]>2000:
        y = y[:2000]
    
    if np.all(np.vectorize(isinstance)(y, str)):
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
    return y

def get_parkinsons_dataset():
    # fetch dataset 
    parkinsons = fetch_ucirepo(id=189) 
    
    # data (as pandas dataframes) 
    X = parkinsons.data.features 
    y = parkinsons.data.targets
    cols = X.columns
    
    X = preprocessing_data_X(X)
    y = preprocessing_data_y(y)
    
    return X, y, cols

def get_performance_data():
    performance = fetch_ucirepo(id=320)
    X = performance.data.features
    y = performance.data.targets
    cols = X.columns
    
    X = preprocessing_data_X(X)
    y = preprocessing_data_y(y)
    
    return X, y, cols

def get_temperature_data():
    temperature = fetch_ucirepo(id=925)
    X = temperature.data.features
    y = temperature.data.targets
    cols = X.columns
    
    X = preprocessing_data_X(X)
    y = preprocessing_data_y(y)
    
    return X, y, cols

def get_ccle_data():
    
    X = pd.read_csv('X_ccle_rnaseq_PD-0325901_top500.csv')
    y = pd.read_csv('y_ccle_rnaseq_PD-0325901.csv')
    cols = X.columns
    X = X.to_numpy()
    y = y.to_numpy().flatten()
    return X, y, cols

def get_adult_dataset(num_samples):
    
    # fetch dataset 
    adult = fetch_ucirepo(id=2) 
    
    # data (as pandas dataframes) 
    X = adult.data.features 
    y = adult.data.targets 

    X = X.dropna()

    # drop the same ones in y, which is a dataframe
    y = y.loc[X.index]
    
    # one hot encode adult dataset
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # convert y to 1 (>50K) and 0 (<=50K)
    y = y.replace({'<=50K' : 0, '<=50K.' : 0, '>50K' : 1, ">50K." : 1})
    y = y['income']
    
    # replace trues and falses in X_encoded with 1s and 0s
    X_encoded = X_encoded.replace({True : 1, False : 0})
    
    # return the first num_samples samples, make all return values numpy arrays
    return X_encoded.iloc[:num_samples].values, y.iloc[:num_samples].values, X_encoded.columns

def split_data(X, y, seed):
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=seed)
    return X_train, X_test, y_train, y_test

def fit_models(X_train, y_train):
    # fit models
    if TASK == "classification":
        rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                    random_state=42)
        rf.fit(X_train, y_train)
        # rf_plus = RandomForestPlusClassifier(rf_model=rf,
        #                                 prediction_model=LogisticRegressionCV())
        rf_plus = RandomForestPlusClassifier(rf_model=rf)
        rf_plus.fit(X_train, y_train)
    elif TASK == "regression":
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                   random_state=42)
        rf.fit(X_train, y_train)
        # rf_plus = RandomForestPlusRegressor(rf_model=rf,
        #                                     prediction_model=RidgeCV())
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
    
    # store command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--datasource', type=str, default=None)
    parser.add_argument('--dataname', type=str, default=None)
    parser.add_argument('--use_test', type=int, default=0)
    parser.add_argument('--njobs', type=int, default=1)
    args = parser.parse_args()
    
    # convert namespace to a dictionary
    args_dict = vars(args)

    # assign the arguments to variables
    seed = args_dict['seed']
    datasource = args_dict['datasource']
    dataname = args_dict['dataname']
    use_test = bool(args_dict['use_test']) # convert from 0/1 to boolean
    njobs = args_dict['njobs']
    
    # if the datasource is openml, we need to make the dataname an integer
    if datasource == "openml":
        dataname = int(dataname)
        
    # if the datasource is a file, we need to read the file rather than call
    # get_clean_dataset
    if datasource == "function":
        if dataname == "adult":
            X, y, feature_names = get_adult_dataset(5000)
        elif dataname == "parkinsons":
            X, y, feature_names = get_parkinsons_dataset()
        elif dataname == "performance":
            X, y, feature_names = get_performance_data()
        elif dataname == "temperature":
            X, y, feature_names = get_temperature_data()
        elif dataname == "ccle":
            X, y, feature_names = get_ccle_data()
        else:
            raise ValueError("Unknown function dataset.")
    else:
        # obtain data
        X, y, feature_names = get_clean_dataset(dataname, data_source = datasource)
        # if y is not a float (abalone), convert it
        if y.dtype != np.float64:
            y = y.astype(np.float64)
    
    # end time
    end = time.time()
    
    # print progress message
    print(f"Progress Message 1/15: Obtained {dataname} from {datasource}.")
    print(f"Step #1 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # check if task is regression or classification
    if len(np.unique(y)) == 2:
        TASK = 'classification'
    else:
        TASK = 'regression'
        # convert y to float, if it is not already (ints will cause errors)
        y = y.astype(float)
        
    # end time
    end = time.time()
    
    print(f"Progress Message 2/15: Task is identified as {TASK}.")
    print(f"Step #2 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # split data
    X_train, X_test, y_train, y_test = split_data(X, y, seed)
    
    # end time
    end = time.time()
    
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")

    print(f"Progress Message 3/15: Data split with seed {seed}.")
    print(f"Step #3 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # fit the prediction models
    rf, rf_plus = fit_models(X_train, y_train)
    
    # fit baseline model
    if TASK == "classification":
        rf_plus_baseline = RandomForestPlusClassifier(rf_model=rf, include_raw=False, fit_on="inbag", prediction_model=LogisticRegression())
    elif TASK == "regression":
        rf_plus_baseline = RandomForestPlusRegressor(rf_model=rf, include_raw=False, fit_on="inbag", prediction_model=LinearRegression())
    rf_plus_baseline.fit(X_train, y_train)
    
    # end time
    end = time.time()
    
    print(f"Progress Message 4/15: RF and RF+ models fit.")
    print(f"Step #4 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # obtain shap feature importances
    shap_explainer = shap.TreeExplainer(rf)
    if use_test:
        shap_values, shap_rankings = get_shap(X_test, shap_explainer)
    else:
        shap_values, shap_rankings = get_shap(X_train, shap_explainer)
    
    # end time
    end = time.time()
    
    print(f"Progress Message 5/15: SHAP values/rankings obtained.")
    print(f"Step #5 took {end-start} seconds.")
    
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
        
    # end time
    end = time.time()
    
    print(f"Progress Message 6/15: LMDI+ values/rankings obtained.")
    print(f"Step #6 took {end-start} seconds.")
    
    # start time
    start = time.time()

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
         # 'normalized_nonl2_ranking': lmdi_values_normalized_nonl2_ranking,
         'baseline': lmdi_values_baseline}
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
        #  'normalized_l2_ranking': lmdi_values_normalized_l2_ranking,
        #  'normalized_nonl2_ranking': lmdi_values_normalized_nonl2_ranking,
         'baseline': lmdi_rankings_baseline}
        
    # get rbo matrices for rankings
    # rbo_matrices = {}
    # for method, ranking in lfi_rankings.items():
    #     for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    #         rbo_matrices[method + "_" + str(p)] = \
    #             compute_rbo_matrix(ranking, 'distance', p=p)
    
    # def compute_rbo_for_method_and_p(method, ranking, p):
    #     """
    #     Helper function to compute the RBO matrix for a given method and p value.
    #     """
    #     # print("method:")
    #     # print(method)
    #     return (method + "_" + str(p), compute_rbo_matrix(ranking, 'distance', p=p))
    
    # # parallelize the computation of RBO matrices
    # rbo_matrices = dict(Parallel(n_jobs=njobs)(
    #     delayed(compute_rbo_for_method_and_p)(method, ranking, p)
    #     for method, ranking in lfi_rankings.items()
    #     for p in [0.1, 0.3, 0.5, 0.7, 0.9]
    # ))
    
    # end time
    end = time.time()

    print(f"Progress Message 7/15: RBO matrices computed.")
    print(f"Step #7 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # get linkages for values
    values_linkage = {}
    for method, values in lfi_values.items():
        # values_linkage[method] = sch.linkage(values, method="ward")
        values_linkage[method] = cluster.hierarchy.ward(values)
        
    # end time
    end = time.time()
        
    print(f"Progress Message 8/15: Linkages for values computed.")
    print(f"Step #8 took {end-start} seconds.")
    
    # start time
    start = time.time()
        
    # # get linkages for rankings
    # rankings_linkage = {}
    # for method, rbo_mat in rbo_matrices.items():
    #     # rankings_linkage[method] = sch.linkage(squareform(rbo_mat),
    #     #                                        method="ward")
    #     rankings_linkage[method] = cluster.hierarchy.ward(squareform(rbo_mat))
        
    # end time
    end = time.time()
        
    print(f"Progress Message 9/15: Linkages for rankings computed.")
    print(f"Step #9 took {end-start} seconds.")
    
    # start time
    start = time.time()

    # get clusters for values
    value_clusters = {}
    for method, link in values_linkage.items():
        # maximum number of clusters is the number of unique feature importances
        max_num_clusters = np.unique(lfi_values[method], axis = 0).shape[0]
        print(f"The Number of Unique Values (Maximum # of Clusters) for {method} is {max_num_clusters}.")
        num_cluster_map = {}
        for num_clusters in np.arange(1, max_num_clusters + 1):
            # num_cluster_map[num_clusters] = fcluster(link, num_clusters,
            #                                          criterion = "maxclust")
            num_cluster_map[num_clusters] = cut_tree(link, n_clusters=num_clusters).flatten()
        value_clusters[method] = num_cluster_map
        
    # end time
    end = time.time()
        
    print(f"Progress Message 10/15: Clusters for values computed.")
    print(f"Step #10 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # # get clusters for rankings
    # ranking_clusters = {}
    # for method, link in rankings_linkage.items():
    #     # maximum number of clusters is the number of unique rankings
    #     max_num_clusters = np.unique(rbo_matrices[method], axis = 0).shape[0]
    #     print(f"The Number of Unique Rankings (Maximum # of Clusters) for {method} is {max_num_clusters}.")
    #     num_cluster_map = {}
    #     for num_clusters in np.arange(1, max_num_clusters + 1):
    #         # num_cluster_map[num_clusters] = fcluster(link, num_clusters,
    #         #                                          criterion = "maxclust")
    #         num_cluster_map[num_clusters] = cut_tree(link, n_clusters=num_clusters).flatten()
    #     ranking_clusters[method] = num_cluster_map
    
    # end time
    end = time.time()
    
    print(f"Progress Message 11/15: Clusters for rankings computed.")
    print(f"Step #11 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # get predictions and performance metrics for each methods clusters
    if TASK == "classification":
        metrics = {"AUROC": roc_auc_score, "AUPRC": average_precision_score,
                   "F1": f1_score, "Accuracy": accuracy_score,
                   "R^2": r2_score, "Cross-Entropy": log_loss}
    elif TASK == "regression":
        metrics = {"R^2": r2_score, "RMSE": root_mean_squared_error}
    
    # maps method to future dataframe (dict for now) where the columns of the
    # dataframe are the metrics and the rows are the number of clusters, and
    # the values are the performance of the method on the metric for the number
    # of clusters.
    # method_values_results = {}
    # for method, cluster_map in value_clusters.items():
    #     metric_results = {}
    #     max_num_clusters = np.max(list(cluster_map.keys()))
    #     metric_results["nclust"] = np.arange(1, max_num_clusters + 1)
    #     for metric, metric_func in metrics.items():
    #         cluster_results = np.repeat(np.nan, max_num_clusters)
    #         for num_clusters, clusters in cluster_map.items():
    #             cluster_predictions = np.repeat(np.nan, len(clusters))
    #             cluster_truths = np.repeat(np.nan, len(clusters))
    #             for i in range(num_clusters):
    #                 cluster_indices = np.where(clusters == i + 1)[0]
    #                 if y_train[cluster_indices].shape[0] == 0:
    #                     continue
    #                 cluster_predictions[cluster_indices] = \
    #                     np.mean(y_train[cluster_indices])
    #                 if metric in ["Accuracy", "F1"]:
    #                     cluster_predictions[cluster_indices] = \
    #                         cluster_predictions[cluster_indices] > 0.5
    #                 cluster_truths[cluster_indices] = y_train[cluster_indices]
    #             cluster_results[num_clusters-1] = metric_func(cluster_truths,
    #                                                         cluster_predictions)
    #         metric_results[metric] = cluster_results
    #     method_values_results[method] = metric_results
    
    def evaluate_method(method, cluster_map, metrics, y_data):
        metric_results = {}
        max_num_clusters = np.max(list(cluster_map.keys()))
        metric_results["nclust"] = np.arange(1, max_num_clusters + 1)
        for metric, metric_func in metrics.items():
            cluster_results = np.repeat(np.nan, max_num_clusters)
            for num_clusters, clusters in cluster_map.items():
                cluster_predictions = np.repeat(np.nan, len(clusters))
                cluster_truths = np.repeat(np.nan, len(clusters))
                for i in range(num_clusters):
                    cluster_indices = np.where(clusters == i)[0]
                    if y_data[cluster_indices].shape[0] == 0:
                        print("ERROR: Empty cluster!")
                        continue
                    cluster_predictions[cluster_indices] = \
                        np.mean(y_data[cluster_indices])
                    if metric in ["Accuracy", "F1"]:
                        cluster_predictions[cluster_indices] = \
                            cluster_predictions[cluster_indices] > 0.5
                    cluster_truths[cluster_indices] = y_data[cluster_indices]
                cluster_results[num_clusters-1] = metric_func(cluster_truths,
                                                            cluster_predictions)
            metric_results[metric] = cluster_results
        return method, metric_results
    
    if use_test:
        method_values_results = dict(Parallel(n_jobs=njobs)(
            delayed(evaluate_method)(method, cluster_map, metrics, y_test)
            for method, cluster_map in value_clusters.items()))
    else:
        method_values_results = dict(Parallel(n_jobs=njobs)(
            delayed(evaluate_method)(method, cluster_map, metrics, y_train)
            for method, cluster_map in value_clusters.items()))
        
    # end time
    end = time.time()
    
    print(f"Progress Message 12/15: Performance metrics computed for values.")
    print(f"Step #12 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # maps method to future dataframe (dict for now) where the columns of the
    # dataframe are the metrics and the rows are the number of clusters, and
    # the values are the performance of the method on the metric for the number
    # of clusters.
    # method_rankings_results = {}
    # for method, cluster_map in ranking_clusters.items():
    #     metric_results = {}
    #     max_num_clusters = np.max(list(cluster_map.keys()))
    #     metric_results["nclust"] = np.arange(1, max_num_clusters + 1)
    #     for metric, metric_func in metrics.items():
    #         cluster_results = np.repeat(np.nan, max_num_clusters)
    #         for num_clusters, clusters in cluster_map.items():
    #             cluster_predictions = np.repeat(np.nan, len(clusters))
    #             cluster_truths = np.repeat(np.nan, len(clusters))
    #             for i in range(num_clusters):
    #                 cluster_indices = np.where(clusters == i + 1)[0]
    #                 if y_train[cluster_indices].shape[0] == 0:
    #                     continue
    #                 cluster_predictions[cluster_indices] = \
    #                     np.mean(y_train[cluster_indices])
    #                 if metric in ["Accuracy", "F1"]:
    #                     cluster_predictions[cluster_indices] = \
    #                         cluster_predictions[cluster_indices] > 0.5
    #                 cluster_truths[cluster_indices] = y_train[cluster_indices]
    #             cluster_results[num_clusters-1] = metric_func(cluster_truths,
    #                                                         cluster_predictions)
    #         metric_results[metric] = cluster_results
    #     method_rankings_results[method] = metric_results
    
    # if use_test:
    #     method_rankings_results = dict(Parallel(n_jobs=njobs)(
    #         delayed(evaluate_method)(method, cluster_map, metrics, y_test)
    #         for method, cluster_map in ranking_clusters.items()))
    # else:
    #     method_rankings_results = dict(Parallel(n_jobs=njobs)(
    #         delayed(evaluate_method)(method, cluster_map, metrics, y_train)
    #         for method, cluster_map in ranking_clusters.items()))
        
    # end time
    end = time.time()
        
    print(f"Progress Message 13/15: Performance metrics computed for rankings.")
    print(f"Step #13 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    if use_test:
        result_dir = oj(os.path.dirname(os.path.realpath(__file__)), 'results/test_data')
    else:
        result_dir = oj(os.path.dirname(os.path.realpath(__file__)), 'results/train_data')
    
    # get result dataframes
    for method, metric_results in method_values_results.items():
        df = pd.DataFrame(metric_results)
        df.to_csv(oj(result_dir,
                     f'{datasource}_{dataname}_seed{seed}_{method}_values.csv'),
                  index=False)
        
    # end time
    end = time.time()
        
    print(f"Progress Message 14/15: Value results saved to {result_dir}.")
    print(f"Step #14 took {end-start} seconds.")
    
    # start time
    start = time.time()
    
    # for method, metric_results in method_rankings_results.items():
    #     df = pd.DataFrame(metric_results)
    #     df.to_csv(oj(result_dir,
    #                 f'{datasource}_{dataname}_seed{seed}_{method}_ranking.csv'),
    #               index=False)
        
    # end time
    end = time.time()
    
    print(f"Progress Message 15/15: Ranking results saved to {result_dir}.")
    print(f"Step #15 took {end-start} seconds.")