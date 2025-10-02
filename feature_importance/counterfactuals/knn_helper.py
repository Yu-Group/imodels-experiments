# sklearn imports
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# data science imports
import numpy as np

# imodels imports
from imodels.tree.rf_plus.rf_plus.rf_plus_models import \
    RandomForestPlusClassifier, RandomForestPlusRegressor
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import LMDIPlus

# data getters
from ucimlrepo import fetch_ucirepo
import openml

# local feature importance
import shap
import lime
from local_mdi import local_mdi_score

# file system
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

def fit_rf_models(X_train, y_train):
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
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=1,
                                max_features='sqrt', random_state=42)
    rf.fit(X_train, y_train)

    # elastic net rf+
    rf_plus = RandomForestPlusClassifier(rf_model=rf,
                prediction_model=LogisticRegressionCV(penalty='elasticnet',
                    l1_ratios=[0.1,0.5,0.99], solver='saga', cv=3,
                    n_jobs=-1, tol=5e-4, max_iter=2000, random_state=42))
    rf_plus.fit(X_train, y_train)

    return rf, rf_plus

def fit_gb_models(X_train, y_train):
    
    gb = GradientBoostingClassifier(n_estimators=100, min_samples_leaf=5,
                                    max_features=0.33, random_state=42)
    gb.fit(X_train, y_train)
    
    # elastic net gb+
    gb_plus_elastic = RandomForestPlusRegressor(rf_model=gb,
                                    prediction_model=ElasticNetCV(cv=3,
                                    l1_ratio=[0.1,0.5,0.99],
                                    max_iter=2000,random_state=42))
    gb_plus_elastic.fit(X_train, y_train)
    
    return gb, gb_plus_elastic

def get_predictions(X, ensemble, ensemble_plus):
    """
    Get the predictions for the given data.
    
    Inputs:
    - X (np.ndarray): The feature matrix.
    - ensemble: The fitted RF/GB object.
    - ensemble_plus: The fitted RF+/GB+ object.
    
    Outputs:
    - ensemble_predictions (np.ndarray): The predictions from the RF/GB model.
    - ensemble_plus_predictions (np.ndarray): The predictions from the RF+/GB+ model.
    """

    ensemble_predictions = ensemble.predict(X)
    ensemble_plus_predictions = ensemble_plus.predict(X)

    return ensemble_predictions, ensemble_plus_predictions

def get_lime(X: np.ndarray, ensemble, is_boosting: bool):
    """
    Get the LIME values and rankings for the given data.
    
    Inputs:
    - X (np.ndarray): The feature matrix.
    - ensemble: The fitted RF/GB object.
    - is_boosting (bool): Whether the model is a boosting model (True) or a random forest model (False).
    
    Outputs:
    - lime_values (np.ndarray): The LIME values.
    """
    
    if is_boosting:
        mode = "regression"
    else:
        mode = "classification"

    lime_values = np.zeros((X.shape[0], X.shape[1]))
    explainer = lime.lime_tabular.LimeTabularExplainer(X, verbose = False,
                                                       mode = mode)
    num_features = X.shape[1]
    for i in range(X.shape[0]):
        if mode == "regression":
            exp = explainer.explain_instance(X[i, :], ensemble.predict,
                                         num_features = num_features)
        else:
            exp = explainer.explain_instance(X[i, :], ensemble.predict_proba,
                                         num_features = num_features)
        original_feature_importance = exp.as_map()[1]
        sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
        for j in range(num_features):
            lime_values[i, j] = sorted_feature_importance[j][1]
        
    return lime_values

def get_shap(X, ensemble, is_boosting: bool):
    """
    Get the SHAP values for the given data.
    
    Inputs:
    - X (np.ndarray): The feature matrix.
    - rf (RandomForestClassifier/Regressor): The fitted RF object.
    
    Outputs:
    - shap_values (np.ndarray): The SHAP values.
    """
    
    shap_explainer = shap.TreeExplainer(ensemble)
    # check if first tree is regression or classification
    if is_boosting:
        shap_values = shap_explainer.shap_values(X, check_additivity=False)
    else:
        shap_values = shap_explainer.shap_values(X, check_additivity=False)[:, :, 1]

    return shap_values

def get_lmdi_plus(X, y, ensemble_plus, inbag=False):
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
        mdi_explainer = LMDIPlus(ensemble_plus, evaluate_on='inbag')
    else:
        mdi_explainer = LMDIPlus(ensemble_plus, evaluate_on='all')
    lmdi_values = mdi_explainer.get_lmdi_plus_scores(X, y, ranking=False)
    
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
    
    if metric == "l1":
        metric = 1
    elif metric == "l2":
        metric = 2
    else:
        raise ValueError("metric must be either 'l1' or 'l2'")
    
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

def perform_rf_pipeline(k, data_id, nbr_dist, cfact_dist, use_preds):
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
    X = np.loadtxt(oj("data", f"{data_id}", "X.csv"), delimiter=",", dtype=float)
    y = np.loadtxt(oj("data", f"{data_id}", "y.csv"), delimiter=",", dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    print("Data Retrieved")
    
    # get fit models
    rf, rf_plus = fit_rf_models(X_train, y_train)

    # mdi_vals = rf.feature_importances_
    
    if use_preds:
        rf_y_test, rf_plus_y_test = \
            get_predictions(X_test, rf, rf_plus)
    
    print("Models Fit")
    
    # get raw data
    raw_train = X_train
    raw_test = X_test
    
    # get shap
    shap_train = get_shap(X_train, rf, is_boosting=False)
    shap_test = get_shap(X_test, rf, is_boosting=False)

    # get lime
    lime_train = get_lime(X_train, rf, is_boosting=False)
    lime_test = get_lime(X_test, rf, is_boosting=False)

    # get lmdi
    lmdi_train, lmdi_test = local_mdi_score(X_train, X_test, model=rf, absolute=False)
    
    # get lmdi plus values
    lmdi_plus_train = get_lmdi_plus(X_train, y_train, rf_plus)
    if use_preds:
        lmdi_plus_test = get_lmdi_plus(X_test, rf_plus_y_test, rf_plus)
    else:
        lmdi_plus_test = get_lmdi_plus(X_test, y_test, rf_plus)

    print("LFI Values Retrieved")
    
    if use_preds:
        raw_opposite = get_k_opposite_neighbors(k, nbr_dist, raw_train, raw_test, y_train, rf_y_test)
        shap_opposite = get_k_opposite_neighbors(k, nbr_dist, shap_train, shap_test, y_train, rf_y_test)
        lime_opposite = get_k_opposite_neighbors(k, nbr_dist, lime_train, lime_test, y_train, rf_y_test)
        lmdi_opposite = get_k_opposite_neighbors(k, nbr_dist, lmdi_train, lmdi_test, y_train, rf_y_test)
        lmdi_plus_opposite = get_k_opposite_neighbors(k, nbr_dist, lmdi_plus_train, lmdi_plus_test, y_train, rf_plus_y_test)
    else:
        raw_opposite = get_k_opposite_neighbors(k, nbr_dist, raw_train, raw_test, y_train, y_test)
        shap_opposite = get_k_opposite_neighbors(k, nbr_dist, shap_train, shap_test, y_train, y_test)
        lime_opposite = get_k_opposite_neighbors(k, nbr_dist, lime_train, lime_test, y_train, y_test)
        lmdi_opposite = get_k_opposite_neighbors(k, nbr_dist, lmdi_train, lmdi_test, y_train, y_test)
        lmdi_plus_opposite = get_k_opposite_neighbors(k, nbr_dist, lmdi_plus_train, lmdi_plus_test, y_train, y_test)

    print(f"Opposite Neighbors Found Using '{nbr_dist}' Distance")
    
    raw_distances = get_average_nbr_dist(k, cfact_dist, raw_opposite, X_train, X_test)
    shap_distances = get_average_nbr_dist(k, cfact_dist, shap_opposite, X_train, X_test)
    lime_distances = get_average_nbr_dist(k, cfact_dist, lime_opposite, X_train, X_test)
    lmdi_distances = get_average_nbr_dist(k, cfact_dist, lmdi_opposite, X_train, X_test)
    lmdi_plus_distances = get_average_nbr_dist(k, cfact_dist, lmdi_plus_opposite, X_train, X_test)

    print(f"Average Distances Calculated")

    return raw_distances, shap_distances, lime_distances, lmdi_distances, lmdi_plus_distances

def perform_gb_pipeline(k, data_id, nbr_dist, cfact_dist, use_preds):
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
    X = np.loadtxt(oj("data", f"{data_id}", "X.csv"), delimiter=",", dtype=float)
    y = np.loadtxt(oj("data", f"{data_id}", "y.csv"), delimiter=",", dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    print("Data Retrieved")
    
    # get fit models
    gb, gb_plus = fit_gb_models(X_train, y_train)

    # mdi_vals = rf.feature_importances_
    
    if use_preds:
        gb_y_test, gb_plus_y_test = \
            get_predictions(X_test, gb, gb_plus)
    
    print("Models Fit")
    
    # get raw data
    raw_train = X_train
    raw_test = X_test
    
    # get shap
    shap_train = get_shap(X_train, gb, is_boosting=True)
    shap_test = get_shap(X_test, gb, is_boosting=True)

    # get lime
    lime_train = get_lime(X_train, gb, is_boosting=True)
    lime_test = get_lime(X_test, gb, is_boosting=True)
    
    # get lmdi plus values
    lmdi_plus_train = get_lmdi_plus(X_train, y_train, gb_plus)
    if use_preds:
        lmdi_plus_test = get_lmdi_plus(X_test, gb_plus_y_test, gb_plus)
    else:
        lmdi_plus_test = get_lmdi_plus(X_test, y_test, gb_plus)

    print("LFI Values Retrieved")
    
    if use_preds:
        raw_opposite = get_k_opposite_neighbors(k, nbr_dist, raw_train, raw_test, y_train, gb_y_test)
        shap_opposite = get_k_opposite_neighbors(k, nbr_dist, shap_train, shap_test, y_train, gb_y_test)
        lime_opposite = get_k_opposite_neighbors(k, nbr_dist, lime_train, lime_test, y_train, gb_y_test)
        lmdi_plus_opposite = get_k_opposite_neighbors(k, nbr_dist, lmdi_plus_train, lmdi_plus_test, y_train, gb_plus_y_test)
    else:
        raw_opposite = get_k_opposite_neighbors(k, nbr_dist, raw_train, raw_test, y_train, y_test)
        shap_opposite = get_k_opposite_neighbors(k, nbr_dist, shap_train, shap_test, y_train, y_test)
        lime_opposite = get_k_opposite_neighbors(k, nbr_dist, lime_train, lime_test, y_train, y_test)
        lmdi_plus_opposite = get_k_opposite_neighbors(k, nbr_dist, lmdi_plus_train, lmdi_plus_test, y_train, y_test)

    print(f"Opposite Neighbors Found Using '{nbr_dist}' Distance")
    
    raw_distances = get_average_nbr_dist(k, cfact_dist, raw_opposite, X_train, X_test)
    shap_distances = get_average_nbr_dist(k, cfact_dist, shap_opposite, X_train, X_test)
    lime_distances = get_average_nbr_dist(k, cfact_dist, lime_opposite, X_train, X_test)
    lmdi_plus_distances = get_average_nbr_dist(k, cfact_dist, lmdi_plus_opposite, X_train, X_test)

    print(f"Average Distances Calculated")

    return raw_distances, shap_distances, lime_distances, lmdi_plus_distances