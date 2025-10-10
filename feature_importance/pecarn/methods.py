# sklearn imports
from sklearn.linear_model import LogisticRegressionCV, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# data science imports
import numpy as np
import pandas as pd

# imodels imports
from imodels import get_clean_dataset
from imodels.tree.rf_plus.rf_plus.rf_plus_models import \
    RandomForestPlusClassifier, RandomForestPlusRegressor
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import LMDIPlus

# local feature importance
import shap
import lime

def get_data() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Fetches the IAI dataset from PECARN.
    
    Returns:
    X (pd.DataFrame): The feature matrix.
    y (np.ndarray): The target vector.
    """
    
    X, y, colnames = get_clean_dataset("iai_pecarn_pred")
    
    X = pd.DataFrame(X, columns=colnames)
    
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
    """
    Fits a GradientBoostingClassifier and a GB+ to the training data.
    
    Parameters:
    X_train (np.ndarray): The training feature matrix.
    y_train (np.ndarray): The training target vector.
    
    Returns:
    gb (GradientBoostingClassifier): The fitted GradientBoostingClassifier.
    gb_plus_elastic (RandomForestPlusClassifier): The fitted GB+.
    """
    
    gb = GradientBoostingClassifier(n_estimators=100, min_samples_leaf=5,
                                    max_depth=None, max_features='sqrt',
                                    random_state=42)
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

# --------------------- Local MDI --------------------- #

def compute_mdi_local_tree(tree, X, vimp):
    nsamples, nfeatures = X.shape

    impurity = tree.impurity
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    features = tree.feature

    for i in range(nsamples):
        node = 0
        oldvimp = impurity[node]

        while children_left[node] != -1:
            ifeat = features[node]
            if X[i, ifeat] <= threshold[node]:
                node = children_left[node]
            else:
                node = children_right[node]
            newvimp = impurity[node]
            vimp[i, ifeat] += oldvimp - newvimp
            oldvimp = newvimp


def compute_mdi_local_ens(ens, X, verbose=0):
    nsamples, nfeatures = X.shape
    # vimp = np.zeros((nsamples, ens.n_features_), dtype='float64')
    vimp = np.zeros((nsamples, nfeatures), dtype='float64')

    for i, est in enumerate(ens.estimators_):
        if verbose > 0:
            print("o", end='', flush=True)
        compute_mdi_local_tree(est.tree_, X, vimp)

    if verbose > 0:
        print("")

    vimp /= ens.n_estimators
    return vimp

def local_mdi_score(X_train, X_test, model=None, absolute=True):
    lfi_train = compute_mdi_local_ens(model, X_train)
    lfi_test = compute_mdi_local_ens(model, X_test)
    if absolute:
        return np.abs(lfi_train), np.abs(lfi_test)
    else:
        return lfi_train, lfi_test