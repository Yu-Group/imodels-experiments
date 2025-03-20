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
        dataset = openml.datasets.get_dataset(data_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        X = X.to_numpy()
        y = y.to_numpy().flatten()
        
        if data_id == 43:
            # transform y from 1/2 to 0/1
            y = (y == 2).astype(int)

    # center and scale the covariates
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

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

    return rf, rf_plus

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

def get_lmdi(X, y, rf_plus):
    """
    Get the LMDI values for the given data.
    
    Inputs:
    - X (np.ndarray): The feature matrix.
    - y (np.ndarray): The target vector.
    - rf_plus (RandomForestPlusClassifier): The fitted RandomForestPlusClassifier.
    
    Outputs:
    - lmdi_values (np.ndarray): The LMDI values.
    """
    
    mdi_explainer = RFPlusMDI(rf_plus, mode="only_k", evaluate_on='all')
    lmdi_values = mdi_explainer.explain_linear_partial(X, y, normalize=False,
                                                       square=False,
                                                       ranking=False)
    
    return lmdi_values

def get_k_opposite_neighbors(k, lfi_valid, lfi_test,
                             y_valid, y_test):
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
    
    # fit nearest neighbors model
    nbrs = NearestNeighbors(n_neighbors=len(lfi_valid))
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

def get_average_nbr_dist(k, lfi_opposite, X_valid, X_test):
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
            distances.append(np.linalg.norm(X_test[i] - X_valid[lfi_opposite[i][j]]))
        lfi_distances.append(distances)
    lfi_distances = np.array(lfi_distances)
    lfi_distances = lfi_distances.mean(axis=1)
    return lfi_distances