# sklearn imports
from sklearn.neighbors import NearestNeighbors, KNeighborsTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# data science imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# imodels imports
from imodels import get_clean_dataset
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusClassifier
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import RFPlusMDI

# data getters
from ucimlrepo import fetch_ucirepo
import openml

# local feature importance
import shap
import lime

# file system
import os
from os.path import join as oj


def get_data(data_source, data_id):
    """
    Fetches dataset from either UCI Machine Learning Repository or OpenML.
    
    Parameters:
    data_source (str): The source of the dataset, either 'uci' or 'openml'.
    data_id (int): The ID of the dataset.
    
    Returns:
    X (np.ndarray): The feature matrix.
    y (np.ndarray): The target vector.
    """
    
    # ensure that data source is either 'uci' or 'openml'
    if data_source not in ["uci", "openml"]:
        raise ValueError("data_source must be either 'uci' or 'openml'")
    
    # handle case where data comes from uci
    if data_source == "uci":
        
        # get pandas df X and numpy array y
        dataset = fetch_ucirepo(id=data_id)
        X = dataset.data.features
        y = dataset.data.targets.to_numpy().flatten()
        
        # handle breast cancer dataset
        if data_id == 15:
            # remove rows with 'nan' entries for 'Bare_nuclei'
            X = X.dropna()
            # remove same observations from dataframe y
            y = y[X.index]
            # reset index
            X = X.reset_index(drop=True)
            # transform y from 2/4 to 0/1
            y = (y == 4).astype(int)
        
        X = X.to_numpy() # convert to numpy

    if data_source == "openml":
        
        # get data
        task = openml.tasks.get_task(data_id)
        dataset = task.get_dataset()
        X, y, categorical_mask, col_names = \
            dataset.get_data(target=dataset.default_target_attribute,
                            dataset_format="array")
        
    # center and scale the covariates
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # sample 2000 rows of X and y if X has more than 2000 rows
    if X.shape[0] > 2000:
        np.random.seed(42)
        indices = np.random.choice(X.shape[0], 2000, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y

def fit_models(X_train, y_train):
    """
    Fits a RandomForestClassifier and a RandomForestPlusClassifier to the training data.
    
    Parameters:
    X_train (np.ndarray): The training feature matrix.
    y_train (np.ndarray): The training target vector.
    
    Returns:
    rf (RandomForestClassifier): The fitted RandomForestClassifier.
    rf_plus_elastic (RandomForestPlusClassifier): The fitted RandomForestPlusClassifier.
    """
    
    # fit random forest
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3,
                                max_features='sqrt', random_state=42)
    rf.fit(X_train, y_train)

    # elastic net rf+
    rf_plus = RandomForestPlusClassifier(rf_model=rf,
                prediction_model=LogisticRegressionCV(penalty='elasticnet',
                    l1_ratios=[0.1,0.5,0.9,0.99], solver='saga', cv=3,
                    n_jobs=-1, tol=5e-4, max_iter=5000, random_state=42))
    rf_plus.fit(X_train, y_train)
    
    # elastic net rf+ with no raw feature
    rf_plus_baseline = RandomForestPlusClassifier(rf_model=rf,
                include_raw=False, fit_on="inbag",
                prediction_model=LogisticRegression(penalty=None))
    rf_plus_baseline.fit(X_train, y_train)

    return rf, rf_plus, rf_plus_baseline

def get_predictions(X, rf, rf_plus, rf_plus_baseline):
    """
    Get the predictions for the given data.
    
    Inputs:
    - X (np.ndarray): The feature matrix.
    - rf (RandomForestClassifier/Regressor): The fitted RF object.
    - rf_plus (RandomForestPlusClassifier): The fitted RandomForestPlusClassifier.
    
    Outputs:
    - rf_predictions (np.ndarray): The predictions from the RF model.
    - rf_plus_predictions (np.ndarray): The predictions from the RF+ model.
    """
    
    rf_predictions = rf.predict(X)
    rf_plus_predictions = rf_plus.predict(X)
    rf_plus_baseline_predictions = rf_plus_baseline.predict(X)
    
    return rf_predictions, rf_plus_predictions, rf_plus_baseline_predictions

def get_lime(X: np.ndarray, rf):
    """
    Get the LIME values and rankings for the given data.
    
    Inputs:
    - X (np.ndarray): The feature matrix.
    - rf (RandomForestClassifier/Regressor): The fitted RF object.
    
    Outputs:
    - lime_values (np.ndarray): The LIME values.
    """
    
    lime_values = np.zeros((X.shape[0], X.shape[1]))
    explainer = lime.lime_tabular.LimeTabularExplainer(X, verbose = False,
                                                       mode = "classification")
    num_features = X.shape[1]
    for i in range(X.shape[0]):
        exp = explainer.explain_instance(X[i, :], rf.predict_proba,
                                         num_features = num_features)
        original_feature_importance = exp.as_map()[1]
        sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
        for j in range(num_features):
            lime_values[i, j] = sorted_feature_importance[j][1]
        
    return lime_values

def get_shap(X, rf):
    """
    Get the SHAP values for the given data.
    
    Inputs:
    - X (np.ndarray): The feature matrix.
    - rf (RandomForestClassifier/Regressor): The fitted RF object.
    
    Outputs:
    - shap_values (np.ndarray): The SHAP values.
    """
    
    shap_explainer = shap.TreeExplainer(rf)
    shap_values = shap_explainer.shap_values(X, check_additivity=False)[:, :, 1]
    
    return shap_values

def get_lmdi(X, y, rf_plus, inbag=False):
    """
    Get the LMDI values for the given data.
    
    Inputs:
    - X (np.ndarray): The feature matrix.
    - y (np.ndarray): The target vector.
    - rf_plus (RandomForestPlusClassifier): The fitted RandomForestPlusClassifier.
    
    Outputs:
    - lmdi_values (np.ndarray): The LMDI values.
    """
    
    if inbag:
        mdi_explainer = RFPlusMDI(rf_plus, mode="only_k", evaluate_on='inbag')
    else:
        mdi_explainer = RFPlusMDI(rf_plus, mode="only_k", evaluate_on='all')
    lmdi_values = mdi_explainer.explain_linear_partial(X, y, normalize=False,
                                                       square=False,
                                                       ranking=False)
    
    return lmdi_values

def get_k_opposite_neighbors(k, metric, lfi_valid, lfi_test, y_valid, y_test): #, weight=False, X_valid=None, X_test=None):
    """
    Find the k closest neighbors to each point in lfi_test that have the opposite label.
    
    Inputs:
    - k (int): The number of neighbors to find.
    - lfi_valid (np.ndarray): The local feature importance values for the validation set.
    - lfi_test (np.ndarray): The local feature importance values for the test set.
    - X_valid (np.ndarray): The validation feature matrix.
    - X_test (np.ndarray): The test feature matrix.
    
    Outputs:
    - opposite_neighbors (list of np.ndarray): The indices of the k closest neighbors with opposite labels for each point in lfi_test.
    """
    
    # if weight is true, then check that X_valid and X_test are provided
    # if weight and (X_valid is None or X_test is None):
    #     raise ValueError("If weight is True, X_valid and X_test must be provided")
    
    if metric == "l1":
        metric = 1
    elif metric == "l2":
        metric = 2
    # elif metric == 'linfty':
    #     metric == "chebyshev"
    else:
        raise ValueError("metric must be either 'l1' or 'l2'")
    
    # fit nearest neighbors model
    # if weight:
    #     avg_lfi_valid = np.mean(np.abs(lfi_valid), axis=0)
    #     nbrs = NearestNeighbors(n_neighbors=len(X_valid), p=metric, metric_params={'w': avg_lfi_valid})
    #     nbrs.fit(X_valid)
    #     lfi_dist, lfi_idxs = nbrs.kneighbors(X_test)
    # else:
    nbrs = NearestNeighbors(n_neighbors=len(lfi_valid), p=metric)
    nbrs.fit(lfi_valid)
    
    # rank points in lfi_valid by distance to each point in lfi_test
    lfi_dist, lfi_idxs = nbrs.kneighbors(lfi_test)
    
    # find the k closest neighbors to each point in lfi_test
    # that have the opposite label
    lfi_opposite = []
    for i in range(lfi_test.shape[0]):
        if y_test[i] == 1:
            opposite = np.where(y_valid == 0)[0]
        else:
            opposite = np.where(y_valid == 1)[0]
        distances = lfi_dist[i][np.isin(lfi_idxs[i], opposite)]
        closest = np.argsort(distances)[:k]
        lfi_opposite.append(lfi_idxs[i][np.isin(lfi_idxs[i], opposite)][closest])
    lfi_opposite = np.array(lfi_opposite)  
    
    return lfi_opposite

def get_average_nbr_dist(k, metric, lfi_opposite, X_valid, X_test):
    """
    Calculate the average distance to the k closest neighbors with opposite labels for each point in lfi_test.
    
    Inputs:
    - k (int): The number of neighbors to consider.
    - lfi_opposite (list of np.ndarray): The indices of the k closest neighbors with opposite labels for each point in lfi_test.
    - X_valid (np.ndarray): The validation feature matrix.
    - X_test (np.ndarray): The test feature matrix.
    
    Outputs:
    - lfi_distances (np.ndarray): The average distances to the k closest neighbors with opposite labels for each point in lfi_test.
    """
    
    if metric == "l1":
        metric = 1
    elif metric == "l2":
        metric = 2
    elif metric == 'chebyshev':
        metric = float("-inf")
    else:
        raise ValueError
    # else:
    #     raise ValueError("metric must be either 'l1', 'l2', or 'linfty'")
    
    lfi_distances = []
    for i in range(X_test.shape[0]):
        distances = []
        for j in range(k):
            distances.append(np.linalg.norm(X_test[i] - X_valid[lfi_opposite[i][j]], ord=metric))
        lfi_distances.append(distances)
    lfi_distances = np.array(lfi_distances)
    lfi_distances = lfi_distances.mean(axis=1)
    return lfi_distances

def get_coord_nbr_dist(k, lfi_opposite, X_valid, X_test):
    """
    Calculate the average distance to the k closest neighbors with opposite labels for each point in lfi_test.
    
    Inputs:
    - k (int): The number of neighbors to consider.
    - lfi_opposite (list of np.ndarray): The indices of the k closest neighbors with opposite labels for each point in lfi_test.
    - X_valid (np.ndarray): The validation feature matrix.
    - X_test (np.ndarray): The test feature matrix.
    
    Outputs:
    - lfi_distances (np.ndarray): The average distances to the k closest neighbors with opposite labels for each point in lfi_test.
    """
    
    lfi_distances = []
    for i in range(X_test.shape[0]):
        distances = []
        for j in range(k):
            distances.append(np.abs(X_test[i] - X_valid[lfi_opposite[i][j]]))
        lfi_distances.append(distances)
    lfi_distances = np.array(lfi_distances)
    lfi_distances = lfi_distances.mean(axis=1)
    return lfi_distances

def perform_pipeline(k, data_id, nbr_dist, cfact_dist, use_preds, weight_by_imp=False, coord_dist=False):
    """
    Perform the entire pipeline of fetching data, fitting models, calculating LFI values,
    finding opposite neighbors, and calculating distances.
    
    Inputs:
    - k (int): The number of neighbors to consider.
    - data_source (str): The source of the dataset, either 'uci' or 'openml'.
    - data_id (int): The ID of the dataset.
    - nbr_dist (str): The distance metric to use for finding neighbors.
    - cfact_dist (str): The distance metric to use for calculating distances.
    
    Outputs:
    - shap_distances (dict): The average distances for SHAP values.
    - lime_distances (dict): The average distances for LIME values.
    - lmdi_distances (dict): The average distances for LMDI values.
    """
    
    # set seed
    np.random.seed(42)
    
    # get and split data
    # X, y, = get_data(data_source, data_id)
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.33, random_state=42)
    # X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    X = np.loadtxt(oj("data", f"{data_id}", "X.csv"), delimiter=",", dtype=float)
    y = np.loadtxt(oj("data", f"{data_id}", "y.csv"), delimiter=",", dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    print("Data Retrieved")
    
    # get fit models
    rf, rf_plus, rf_plus_baseline = fit_models(X_train, y_train)
    
    mdi_vals = rf.feature_importances_
    
    if use_preds:
        rf_y_test, rf_plus_y_test, rf_plus_baseline_y_test = \
            get_predictions(X_test, rf, rf_plus, rf_plus_baseline)
    
    print("Models Fit")
    
    # get raw data
    raw_train = X_train
    raw_test = X_test
    
    # get shap
    shap_train = get_shap(X_train, rf)
    shap_test = get_shap(X_test, rf)

    # get lime
    lime_train = get_lime(X_train, rf)
    lime_test = get_lime(X_test, rf)
    
    # get lmdi values
    lmdi_train = get_lmdi(X_train, y_train, rf_plus)
    if use_preds:
        lmdi_test = get_lmdi(X_test, rf_plus_y_test, rf_plus)
    else:
        lmdi_test = get_lmdi(X_test, y_test, rf_plus)
        
    # get lmdi baseline values
    lmdi_baseline_train = get_lmdi(X_train, y_train, rf_plus_baseline)
    if use_preds:
        lmdi_baseline_test = get_lmdi(X_test, rf_plus_baseline_y_test, rf_plus_baseline)
    else:
        lmdi_baseline_test = get_lmdi(X_test, y_test, rf_plus_baseline)
    
    print("LFI Values Retrieved")
    
    # metrics = ["l1", "l2", "linfty"]
    
    # shap_opposite = {}
    # lime_opposite = {}
    # lmdi_opposite = {}
    
    # for metric in metrics:
    #     print(f"Calculating neighbors for metric: {metric}")
    #     shap_opposite[metric] = get_k_opposite_neighbors(k, metric, shap_valid, shap_test, y_valid, y_test)
    #     lime_opposite[metric] = get_k_opposite_neighbors(k, metric, lime_valid, lime_test, y_valid, y_test)
    #     lmdi_opposite[metric] = get_k_opposite_neighbors(k, metric, lmdi_valid, lmdi_test, y_valid, y_test)
    
    if weight_by_imp:
        if use_preds:
            raw_opposite = None # get_k_opposite_neighbors(k, nbr_dist, raw_train, raw_test, y_train, rf_y_test)
            shap_opposite = get_k_opposite_neighbors(k, nbr_dist, shap_train, shap_test, y_train, rf_y_test, weight=True, X_valid=X_train, X_test=X_test)
            lime_opposite = get_k_opposite_neighbors(k, nbr_dist, lime_train, lime_test, y_train, rf_y_test, weight=True, X_valid=X_train, X_test=X_test)
            lmdi_opposite = get_k_opposite_neighbors(k, nbr_dist, lmdi_train, lmdi_test, y_train, rf_plus_y_test, weight=True, X_valid=X_train, X_test=X_test)
            lmdi_baseline_opposite = get_k_opposite_neighbors(k, nbr_dist, lmdi_baseline_train, lmdi_baseline_test, y_train, rf_plus_baseline_y_test, weight=True, X_valid=X_train, X_test=X_test)
        else:
            raw_opposite = None # get_k_opposite_neighbors(k, nbr_dist, raw_train, raw_test, y_train, y_test)
            shap_opposite = get_k_opposite_neighbors(k, nbr_dist, shap_train, shap_test, y_train, y_test, weight=True, X_valid=X_train, X_test=X_test)
            lime_opposite = get_k_opposite_neighbors(k, nbr_dist, lime_train, lime_test, y_train, y_test, weight=True, X_valid=X_train, X_test=X_test)
            lmdi_opposite = get_k_opposite_neighbors(k, nbr_dist, lmdi_train, lmdi_test, y_train, y_test, weight=True, X_valid=X_train, X_test=X_test)
            lmdi_baseline_opposite = get_k_opposite_neighbors(k, nbr_dist, lmdi_baseline_train, lmdi_baseline_test, y_train, y_test, weight=True, X_valid=X_train, X_test=X_test)
    else:
        if use_preds:
            raw_opposite = get_k_opposite_neighbors(k, nbr_dist, raw_train, raw_test, y_train, rf_y_test)
            shap_opposite = get_k_opposite_neighbors(k, nbr_dist, shap_train, shap_test, y_train, rf_y_test)
            lime_opposite = get_k_opposite_neighbors(k, nbr_dist, lime_train, lime_test, y_train, rf_y_test)
            lmdi_opposite = get_k_opposite_neighbors(k, nbr_dist, lmdi_train, lmdi_test, y_train, rf_plus_y_test)
            lmdi_baseline_opposite = get_k_opposite_neighbors(k, nbr_dist, lmdi_baseline_train, lmdi_baseline_test, y_train, rf_plus_baseline_y_test)
        else:
            raw_opposite = get_k_opposite_neighbors(k, nbr_dist, raw_train, raw_test, y_train, y_test)
            shap_opposite = get_k_opposite_neighbors(k, nbr_dist, shap_train, shap_test, y_train, y_test)
            lime_opposite = get_k_opposite_neighbors(k, nbr_dist, lime_train, lime_test, y_train, y_test)
            lmdi_opposite = get_k_opposite_neighbors(k, nbr_dist, lmdi_train, lmdi_test, y_train, y_test)
            lmdi_baseline_opposite = get_k_opposite_neighbors(k, nbr_dist, lmdi_baseline_train, lmdi_baseline_test, y_train, y_test)
    
    print(f"Opposite Neighbors Found Using '{nbr_dist}' Distance")
    
    # shap_distances = {}
    # lime_distances = {}
    # lmdi_distances = {}
    
    # for metric1 in metrics:
    #     shap_distances[metric1] = {}
    #     lime_distances[metric1] = {}
    #     lmdi_distances[metric1] = {}
    #     for metric2 in metrics:
    #         print(f"Calculating distances for metric: {metric1} based on neighbors from metric: {metric2}")
    #         shap_distances[metric1][metric2] = get_average_nbr_dist(k, metric1, shap_opposite[metric2], X_valid, X_test)
    #         lime_distances[metric1][metric2] = get_average_nbr_dist(k, metric1, lime_opposite[metric2], X_valid, X_test)
    #         lmdi_distances[metric1][metric2] = get_average_nbr_dist(k, metric1, lmdi_opposite[metric2], X_valid, X_test)

    if coord_dist:
        if weight_by_imp:
            raw_distances = None
        else:
            raw_distances = get_coord_nbr_dist(k, raw_opposite, X_train, X_test)
        shap_distances = get_coord_nbr_dist(k, shap_opposite, X_train, X_test)
        lime_distances = get_coord_nbr_dist(k, lime_opposite, X_train, X_test)
        lmdi_distances = get_coord_nbr_dist(k, lmdi_opposite, X_train, X_test)
        lmdi_baseline_distances = get_coord_nbr_dist(k, lmdi_baseline_opposite, X_train, X_test)
    else:
        if weight_by_imp:
            raw_distances = None
        else:
            raw_distances = get_average_nbr_dist(k, cfact_dist, raw_opposite, X_train, X_test)
        shap_distances = get_average_nbr_dist(k, cfact_dist, shap_opposite, X_train, X_test)
        lime_distances = get_average_nbr_dist(k, cfact_dist, lime_opposite, X_train, X_test)
        lmdi_distances = get_average_nbr_dist(k, cfact_dist, lmdi_opposite, X_train, X_test)
        lmdi_baseline_distances = get_average_nbr_dist(k, cfact_dist, lmdi_baseline_opposite, X_train, X_test)
    
    print(f"Average Distances Calculated")

    # plot distances
    # plt.figure(figsize=(10, 6))
    # plt.hist(shap_distances, bins=30, alpha=0.5, label='SHAP', color='blue')
    # plt.hist(lime_distances, bins=30, alpha=0.5, label='LIME', color='orange')
    # plt.hist(lmdi_distances, bins=30, alpha=0.5, label='LMDI', color='green')
    # plt.xlabel('Average Distance to 3 Closest Opposite Label Neighbors')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Distances to Three Closest Opposite Label Neighbors')
    # plt.legend()
    # plt.show()
    
    if coord_dist:
        return raw_distances, shap_distances, lime_distances, lmdi_distances, lmdi_baseline_distances, mdi_vals, np.abs(shap_test), np.abs(lime_test), np.abs(lmdi_test), np.abs(lmdi_baseline_test)
    else:
        return raw_distances, shap_distances, lime_distances, lmdi_distances, lmdi_baseline_distances
    # return raw_distances, shap_distances, lime_distances, lmdi_distances, mdi_vals, np.abs(shap_train), np.abs(lime_train), np.abs(lmdi_train)