# standard data science packages
import numpy as np
import pandas as pd
from scipy import cluster

# imodels imports
from imodels.tree.rf_plus.rf_plus.rf_plus_models import \
    RandomForestPlusRegressor, RandomForestPlusClassifier
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import \
    RFPlusMDI, AloRFPlusMDI

# functions for subgroup experiments
from subgroup_detection import *
from subgroup_experiment import *
import shap
import lime

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV,\
    RidgeCV, ElasticNetCV, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, \
    accuracy_score, r2_score, f1_score, log_loss, root_mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# for reading data
import openml

# for saving results
import argparse
import os
from os.path import join as oj

# for function headers
from typing import Tuple, Dict

# because openml package has pesky FutureWarnings
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning,
                        module='openml')


def get_openml_data(id: int, standardize: bool,
                    num_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get benchmark datasets from OpenML.
    
    Inputs:
    - id (int): The OpenML dataset ID.
    - num_samples (int): The number of samples to use. If None, use all samples.
    
    Outputs:
    - X (np.ndarray): The feature matrix.
    - y (np.ndarray): The target vector.
    """
    
    # check that the dataset_id is in the set of tested datasets
    regr_ids = {361234, 361235, 361236, 361237, 361241, 361242, 361243, 361244,
                361247, 361249, 361250, 361251, 361252, 361253, 361254, 361255,
                361256, 361257, 361258, 361259, 361260, 361261, 361264, 361266,
                361267, 361268, 361269, 361272, 361616, 361617, 361618, 361619,
                361621, 361622, 361623}
    binary_class_ids = {31, 10101, 3913, 3, 3917, 9957, 9946, 3918, 3903, 37,
                        9971, 9952, 3902, 49, 43, 9978, 10093, 219, 9976, 14965,
                        9977, 15, 29, 14952, 125920, 3904, 9910, 3021, 7592,
                        146820, 146819, 14954, 167141, 167120, 167125}
    
    # error if not recognized, since we do not know if we need to transform
    if id not in regr_ids and id not in binary_class_ids:
        raise ValueError(f"Data ID {id} is not a benchmark dataset.")
        
    # get the dataset from openml
    task = openml.tasks.get_task(id)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    # if the dataset is categorical, we may have to convert its type
    if id in binary_class_ids:
        # check if it is categorical, convert to 1/0
        if pd.api.types.is_categorical_dtype(y):
            y = pd.get_dummies(y, drop_first=True, dtype=float)
            
    
    # one-hot encode the categorical variables
    X = pd.get_dummies(X, dtype=float)
    
    # remove rows with missing values
    missing_rows = X.isnull().any(axis=1)
    X = X[~missing_rows]
    y = y[~missing_rows]
    
    # subsample the data if necessary
    if num_samples is not None and num_samples < X.shape[0]:
        X = X.sample(num_samples, random_state=1)
        y = y.loc[X.index]
    
    # reset the index of X and y
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # convert X and y to numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()
    
    # if y is not a 1D array, convert it to one
    if len(y.shape) > 1:
        y = y.reshape(-1)
    
    # convert y to float (sometimes, it is int, e.g. abalone dataset)
    y = y.astype(float)
    
    # perform transformations if needed
    log_transform = {361236, 361242, 361244, 361252, 361257, 361260, 361261,
                     361264, 361266, 361267, 361272, 361618, 361622}
    if id in log_transform:
        if np.min(y) > 0:
            y = np.log(y)
        if np.min(y) <= 0:
            y = np.log(y - np.min(y) + 1)
    
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        # if the dataset is a regression dataset, standardize y
        if id in regr_ids:
            y = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X, y

def fit_models(X_train: np.ndarray, y_train: np.ndarray, task: str):
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
    if task == "classification":

        # fit random forest with params from MDI+
        rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3,
                                    max_features='sqrt', random_state=42)
        rf.fit(X_train, y_train)

        # baseline rf+ includes no raw feature and only fits on in-bag samples
        rf_plus_baseline = RandomForestPlusClassifier(rf_model=rf,
                                        include_raw=False, fit_on="inbag",
                                        prediction_model=LogisticRegression())
        rf_plus_baseline.fit(X_train, y_train)
        
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

        # standard rf+ uses default values
        # rf_plus = RandomForestPlusClassifier(rf_model=rf)
        # rf_plus.fit(X_train, y_train)

    # if regression, fit regressors
    elif task == "regression":
        
        # fit random forest with params from MDI+
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                   max_features=0.33, random_state=42)
        rf.fit(X_train, y_train)
        
        # baseline rf+ includes no raw feature and only fits on in-bag samples
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
        
        # standard rf+ uses default values
        # rf_plus = RandomForestPlusRegressor(rf_model=rf)
        # rf_plus.fit(X_train, y_train)
    
    # otherwise, throw error
    else:
        raise ValueError("Task must be 'classification' or 'regression'.")
    
    # return tuple of models
    return rf, rf_plus_baseline, rf_plus_ridge, rf_plus_lasso, rf_plus_elastic

def create_lmdi_variant_map() -> Dict[str, Dict[str, bool]]:
    """
    Create a mapping of LMDI+ variants to argument mappings.
    
    Outputs:
    - lmdi_variants (Dict[str, Dict[str, bool]]): The LMDI variants to use.
    """
    
    # enumerate the different options when initializing a LMDI+ explainer.
    glm = ["ridge", "lasso", "elastic"]
    l2norm = {True: "l2", False: "nonl2"}
    sign = {True: "signed", False: "unsigned"}
    normalize = {True: "normed", False: "nonnormed"}
    leaf_average = {True: "leafavg", False: "noleafavg"}
    ranking = {True: "rank", False: "norank"}
    
    # create the mapping of variants to argument mappings
    lmdi_variants = {}
    for g in glm:
        for l2 in l2norm:
            for s in sign:
                for n in normalize:
                    # sign and normalize are only relevant if l2norm is True
                    if (not l2) and (s or n):
                        continue
                    for la in leaf_average:
                        for r in ranking:
                            # ranking is only relevant if leaf_average is False
                            if la:
                                continue
                            # create the name the variant will be stored under
                            variant_name = f"{g}_{l2norm[l2]}_{sign[s]}" + \
                            f"_{normalize[n]}_{leaf_average[la]}_{ranking[r]}"
                            # store the arguments for the lmdi+ explainer
                            arg_map = {"glm": g, "l2norm": l2, "sign": s,
                                       "normalize": n, "leaf_average": la,
                                       "ranking": r}
                            lmdi_variants[variant_name] = arg_map
    
    return lmdi_variants

# def create_lmdi_variant_map() -> Dict[str, Dict[str, bool]]:
#     """
#     Create a mapping of LMDI+ variants to argument mappings.
    
#     Outputs:
#     - lmdi_variants (Dict[str, Dict[str, bool]]): The LMDI variants to use.
#     """
    
#     # enumerate the different options when initializing a LMDI+ explainer.
#     loo = {True: "aloo", False: "nonloo"}
#     l2norm = {True: "l2", False: "nonl2"}
#     sign = {True: "signed", False: "unsigned"}
#     normalize = {True: "normed", False: "nonnormed"}
#     leaf_average = {True: "leafavg", False: "noleafavg"}
#     ranking = {True: "rank", False: "norank"}
    
#     # create the mapping of variants to argument mappings
#     lmdi_variants = {}
#     for l in loo:
#         for n in l2norm:
#             for s in sign:
#                 for nn in normalize:
#                     # sign and normalize are only relevant if l2norm is True
#                     if (not n) and (s or nn):
#                         continue
#                     for la in leaf_average:
#                         for r in ranking:
#                             # ranking is only relevant if leaf_average is False
#                             if la:
#                                 continue
#                             # create the name the variant will be stored under
#                             variant_name = f"{loo[l]}_{l2norm[n]}_{sign[s]}" + \
#                             f"_{normalize[nn]}_{leaf_average[la]}_{ranking[r]}"
#                             # store the arguments for the lmdi+ explainer
#                             arg_map = {"loo": l, "l2norm": n, "sign": s,
#                                        "normalize": nn, "leaf_average": la,
#                                        "ranking": r}
#                             lmdi_variants[variant_name] = arg_map
    
#     return lmdi_variants

def get_shap(X: np.ndarray, shap_explainer: shap.TreeExplainer, task: str):
    """
    Get the SHAP values and rankings for the given data.
    
    Inputs:
    - X (np.ndarray): The feature matrix.
    - shap_explainer (shap.TreeExplainer): The SHAP explainer object.
    - task (str): The task type, either 'classification' or 'regression'.
    
    Outputs:
    - shap_values (np.ndarray): The SHAP values.
    - shap_rankings (np.ndarray): The SHAP rankings.
    """
    
    # classification and regression get indexed differently
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

def get_lmdi_explainers(rf_plus_baseline, rf_plus_ridge,
                        rf_plus_lasso, rf_plus_elastic,
                        lmdi_variants: Dict[str, Dict[str, bool]]):
    """
    Create the LMDI explainer objects for the subgroup experiments.
    
    Inputs:
    - rf_plus_baseline (RandomForestPlusClassifier/RandomForestPlusRegressor):
                    The baseline RF+ (no raw feature, only on in-bag samples,
                    regular linear/logistic prediction model).
    - rf_plus_ridge (RandomForestPlusClassifier/RandomForestPlusRegressor):
                    The RF+ with a ridge prediction model.
    - rf_plus_lasso (RandomForestPlusClassifier/RandomForestPlusRegressor):
                    The RF+ with a lasso prediction model.
    - rf_plus_elastic (RandomForestPlusClassifier/RandomForestPlusRegressor):
                    The RF+ with an elastic net prediction model.
    - lmdi_variants (Dict[str, Dict[str, bool]]): The LMDI variants to use.
                    Stored as a dictionary with keys corresponding to the name
                    of the lmdi+ variant and the value correponding to the
                    argument mapping. In the argument mapping, keys are strings
                    corresponding to elements of the variant (e.g. "loo") and
                    the values are bools to indicate if they have that property.
    
    Outputs:
    - lmdi_explainers (Dict[str, RFPlusMDI/AloRFPlusMDI]): The LMDI+ explainer
                    objects. The keys correspond to the variant names, and the
                    values are the explainer objects, where AloRFPlusMDIs are
                    used if "loo" is True and RFPlusMDIs are used if "loo" is
                    False. Unique objects are used for each variant to prevent
                    saved fields from interfering with results.
    """
    
    lmdi_explainers = {}
    
    # if a baseline is provided, we need to treat it separately
    if rf_plus_baseline is not None:
        # evaluate on inbag samples only
        lmdi_explainers["lmdi_baseline"] = RFPlusMDI(rf_plus_baseline,
                                                     mode = "only_k",
                                                     evaluate_on = "inbag")
    for variant_name in lmdi_variants.keys():
        lmdi_explainers
    
    # create the explainer objects for each variant, using AloRFPlusMDI if loo
    # is True and RFPlusMDI if loo is False
    for variant_name in lmdi_variants.keys():
        if lmdi_variants[variant_name]["glm"] == "ridge":
            lmdi_explainers[variant_name] = RFPlusMDI(rf_plus_ridge,
                                                      mode = "only_k")
        elif lmdi_variants[variant_name]["glm"] == "lasso":
            lmdi_explainers[variant_name] = RFPlusMDI(rf_plus_lasso,
                                                      mode = "only_k")
        elif lmdi_variants[variant_name]["glm"] == "elastic":
            lmdi_explainers[variant_name] = RFPlusMDI(rf_plus_elastic,
                                                      mode = "only_k")
        else:
            raise ValueError("Invalid GLM type.")
    
    return lmdi_explainers

# def get_lmdi_explainers(rf_plus, lmdi_variants: Dict[str, Dict[str, bool]],
#                         rf_plus_baseline = None, rf_plus_lasso = None,
#                         rf_plus_ridge = None):
#     """
#     Create the LMDI explainer objects for the subgroup experiments.
    
#     Inputs:
#     - rf_plus (RandomForestPlusClassifier/RandomForestPlusRegressor): The RF+.
#     - lmdi_variants (Dict[str, Dict[str, bool]]): The LMDI variants to use.
#                     Stored as a dictionary with keys corresponding to the name
#                     of the lmdi+ variant and the value correponding to the
#                     argument mapping. In the argument mapping, keys are strings
#                     corresponding to elements of the variant (e.g. "loo") and
#                     the values are bools to indicate if they have that property.
#     - rf_plus_baseline (RandomForestPlusClassifier/RandomForestPlusRegressor):
#                     The baseline RF+ (no raw feature, only on in-bag samples,
#                     regular linear/logistic prediction model).
#     - rf_plus_lasso (RandomForestPlusClassifier/RandomForestPlusRegressor):
#                     The version of RF+ that worked best for Zhongyuan's feature
#                     selection experiments.
    
#     Outputs:
#     - lmdi_explainers (Dict[str, RFPlusMDI/AloRFPlusMDI]): The LMDI+ explainer
#                     objects. The keys correspond to the variant names, and the
#                     values are the explainer objects, where AloRFPlusMDIs are
#                     used if "loo" is True and RFPlusMDIs are used if "loo" is
#                     False. Unique objects are used for each variant to prevent
#                     saved fields from interfering with results.
#     """
    
#     lmdi_explainers = {}
    
#     # if a baseline is provided, we need to treat it separately
#     if rf_plus_baseline is not None:
#         # evaluate on inbag samples only
#         lmdi_explainers["lmdi_baseline"] = RFPlusMDI(rf_plus_baseline,
#                                                      mode = "only_k",
#                                                      evaluate_on = "inbag")
#     if rf_plus_lasso is not None:
#         lmdi_explainers["lmdi_lasso"] = RFPlusMDI(rf_plus_lasso,
#                                                       mode="only_k",
#                                                       evaluate_on="all")
#     if rf_plus_ridge is not None:
#         lmdi_explainers["lmdi_ridge"] = RFPlusMDI(rf_plus_ridge,
#                                                       mode="only_k",
#                                                       evaluate_on="all")
#     # create the explainer objects for each variant, using AloRFPlusMDI if loo
#     # is True and RFPlusMDI if loo is False
#     for variant_name in lmdi_variants.keys():
#         if lmdi_variants[variant_name]["loo"]:
#             lmdi_explainers[variant_name] = AloRFPlusMDI(rf_plus,
#                                                          mode = "only_k")
#         else:
#             lmdi_explainers[variant_name] = RFPlusMDI(rf_plus, mode = "only_k")
    
#     return lmdi_explainers
    

def get_lmdi(X: np.ndarray, y: np.ndarray,
             lmdi_variants: Dict[str, Dict[str, bool]], lmdi_explainers):
    """
    Get the LMDI+ values and rankings for the given data.
    
    Inputs:
    - X (np.ndarray): The feature matrix.
    - y (np.ndarray): The target vector. For test data, y should be None.
    - lmdi_variants (Dict[str, Dict[str, bool]]): The LMDI variants to use.
                    Stored as a dictionary with keys corresponding to the name
                    of the lmdi+ variant and the value correponding to the
                    argument mapping. In the argument mapping, keys are strings
                    corresponding to elements of the variant (e.g. "loo") and
                    the values are bools to indicate if they have that property.
    - lmdi_explainers (Dict[str, RFPlusMDI/AloRFPlusMDI]): The LMDI+ explainer
                    objects. The keys correspond to the variant names, and the
                    values are the explainer objects, where AloRFPlusMDIs are
                    used if "loo" is True and RFPlusMDIs are used if "loo" is
                    False. Unique objects are used for each variant to prevent
                    saved fields from interfering with results.
    
    Outputs:
    - lmdi_values (Dict[str, np.ndarray]): Mapping with variant names as keys
                    and numpy arrays of the LMDI+ values as values.
    - lmdi_rankings (Dict[str, np.ndarray]): Mapping with variant names as keys
                    and numpy arrays of the LMDI+ rankings as values.
    """
    
    # initialize storage mappings
    lmdi_values = {}
    lmdi_rankings = {}
    
    # if the explainer mapping has a baseline, we need to treat it differently
    if "lmdi_baseline" in lmdi_explainers:
        
        # we need to get the values with all of the params set to False
        lmdi_values["lmdi_baseline"] = \
            lmdi_explainers["lmdi_baseline"].explain_linear_partial(X, y,
                                            l2norm=False, sign=False,
                                            normalize=False, leaf_average=False,
                                            ranking=False)

        # get the rankings using the method in the explainer class
        lmdi_rankings["lmdi_baseline"] = \
            lmdi_explainers["lmdi_baseline"].get_rankings(
                np.abs(lmdi_values["lmdi_baseline"])
                )
    
    # if the explainer mapping has a lasso variant, we treat it differently
    if "lmdi_lasso" in lmdi_explainers:
        # print("Computing LMDI+ for Lasso variant...")
        # lmdi_values["lmdi_lasso"] = \
        #     lmdi_explainers["lmdi_lasso"].explain_linear_partial(X, y,
        #                                     l2norm=True, sign=True,
        #                                     normalize=True, leaf_average=False,
        #                                     ranking=True)
        lmdi_values["lmdi_lasso"] = \
            lmdi_explainers["lmdi_lasso"].explain_linear_partial(X, y,
                                            l2norm=False, sign=False,
                                            normalize=False, leaf_average=False,
                                            ranking=True)
        # print("Done, values are:")
        # print(lmdi_values["lmdi_lasso"])
    
    if "lmdi_ridge" in lmdi_explainers:
        # print("Computing LMDI+ for Ridge variant...")
        # lmdi_values["lmdi_ridge"] = \
        #     lmdi_explainers["lmdi_ridge"].explain_linear_partial(X, y,
        #                                     l2norm=True, sign=True,
        #                                     normalize=True, leaf_average=False,
        #                                     ranking=True)
        lmdi_values["lmdi_ridge"] = \
            lmdi_explainers["lmdi_ridge"].explain_linear_partial(X, y,
                                            l2norm=False, sign=False,
                                            normalize=False, leaf_average=False,
                                            ranking=True)
        # print("Done, values are:")
        # print
    
    # for all the other variants, we loop through the explainer objects,
    # using their parameter mappings to set the arguments.
    for name, explainer in lmdi_explainers.items():
        
        # skip through the baseline model, since we have already done it
        if name == "lmdi_baseline" or name == "lmdi_lasso" or name == "lmdi_ridge":
            continue
        
        # get the argument mapping
        variant_args = lmdi_variants[name]    
        
        # set the values by calling explain on the object with the args from
        # input mapping    
        lmdi_values[name] = explainer.explain_linear_partial(X, y,
                                        l2norm=variant_args["l2norm"],
                                        sign=variant_args["sign"],
                                        normalize=variant_args["normalize"],
                                        leaf_average=variant_args["leaf_average"],
                                        ranking=variant_args["ranking"])
        
        # get rankings using the method in the explainer class. absolute value
        # taken to ensure that the rankings are based on the magnitude.
        lmdi_rankings[name] = explainer.get_rankings(np.abs(lmdi_values[name]))
        
    return lmdi_values, lmdi_rankings

def get_train_clusters(lfi_train_values: Dict[str, np.ndarray], method: str):
    """
    Get the clusters for the training data.
    
    Inputs:
    - lfi_train_values (Dict[str, np.ndarray]): The LMDI+ values for the
                    training data. The keys correspond to the method names and
                    the values are numpy arrays of the LMDI+ values.
    - method (str): The clustering method to use, either 'kmeans' or
                    'hierarchical'.
    
    Outputs:
    - method_to_indices (Dict[str, Dict[int, Dict[int, np.ndarray]]]): Mapping
                    with method names as keys and dictionaries as values. The
                    inner dictionaries have the number of clusters as keys and
                    dictionaries as values. The innermost dictionaries have the
                    cluster numbers as keys and numpy arrays of the indices in
                    each cluster as values.
    Removed:
    - method_to_labels (Dict[str, Dict[int, np.ndarray]]): Mapping with method
                    names as keys and dictionaries as values. The inner
                    dictionaries have the number of clusters as keys and numpy
                    arrays of the cluster assignments as values.
    """
    
    # make sure method is valid
    if method not in ["kmeans", "hierarchical"]:
        raise ValueError("Method must be 'kmeans' or 'hierarchical'.")
    
    # case when hierarchical clustering
    if method == "hierarchical":

        # store the linkage for each variant to compute clusters later
        train_linkage = {}
        for variant, values in lfi_train_values.items():
            # use ward linkage
            train_linkage[variant] = cluster.hierarchy.ward(values)
            
        # store the clusters for each variant
        method_to_labels = {}
        # loop through each variant
        for variant, link in train_linkage.items():
            # each variant will have a mapping to store its clusters.
            # the number of clusters will be the key, and the array of cluster
            # assignments will be the value.
            num_cluster_map = {}
            # number of clusters varies from 2 to 10
            for num_clusters in range(2, 11):
                # cut the tree at the specified number of clusters,
                # saving the flattened array of cluster assignments
                num_cluster_map[num_clusters] = \
                    cluster.hierarchy.cut_tree(link,
                                            n_clusters=num_clusters).flatten()
            # store the mapping for the variant
            method_to_labels[variant] = num_cluster_map
            
    # case when kmeans clustering
    else:
        # store the clusters for each variant
        method_to_labels = {}
        # loop through each variant
        for variant, values in lfi_train_values.items():
            # each variant will have a mapping to store its clusters.
            # the number of clusters will be the key, and the array of cluster
            # assignments will be the value.
            num_cluster_map = {}
            # number of clusters varies from 2 to 10
            for num_clusters in range(2, 11):
                
                ### kmeans - recommended because different random inits
                kmeans = KMeans(n_clusters=num_clusters, random_state=42,
                                n_init=10)
                kmeans.fit(values)
                num_cluster_map[num_clusters] = kmeans.labels_
                
                ### scipy - shouldn't use
                # obtain the cluster centroids first (this is how they suggest
                # doing it, although it feels weird)
                # centroids, _ = cluster.vq.kmeans(obs=values,
                #                                  k_or_guess=num_clusters)
                # assign the values to the clusters using centroids
                # kmeans, _ = cluster.vq.vq(values, centroids)
                # num_cluster_map[num_clusters] = kmeans
            # store the mapping for the variant
            method_to_labels[variant] = num_cluster_map
    
    # at this point, we have method_to_labels, which is a mapping of variant
    # name to another mapping. for each variant, the mapping is from the number
    # of clusters c to the cluster assignments when there are c clusters.
    # HOWEVER, what we want is a mapping that eventually goes from the variant
    # name to the indices in each cluster. thus, we need to convert the cluster
    # labels to the indexes that fall into each cluster.
    
    method_to_indices = {}
    for variant, clusters in method_to_labels.items():
        # this will be a mapping from the number of clusters to another mapping,
        # which will be from the cluster number to the indices in that cluster.
        num_cluster_map = {}
        for num_clusters, cluster_labels in clusters.items():
            # this is the inner mapping mentioned above
            cluster_map = {}
            for cluster_num in range(num_clusters):
                # for each cluster, get the indices with that cluster label
                cluster_indices = np.where(cluster_labels == cluster_num)[0]
                cluster_map[cluster_num] = cluster_indices
            num_cluster_map[num_clusters] = cluster_map
        method_to_indices[variant] = num_cluster_map
        
    # we need to return both versions because I don't want to have to rewrite
    # functions that rely on different versions of the mapping.
    # return method_to_labels, method_to_indices
    
    return method_to_indices

# def get_cluster_centroids(lfi_train_values: Dict[str, np.ndarray],
#                           method_to_labels: Dict[str, Dict[int, np.ndarray]]):
#     """
#     Gets the middle of each cluster so that test data can be assigned.
    
#     Inputs:
#     - lfi_train_values (Dict[str, np.ndarray]): The LMDI+ values for the
#                     training data. The keys correspond to the method names and
#                     the values are numpy arrays of the LMDI+ values.
#     - method_to_labels (Dict[str, Dict[int, np.ndarray]]): Mapping with method
#                     names as keys and dictionaries as values. The inner
#                     dictionaries have the number of clusters as keys and numpy
#                     arrays of the cluster assignments as values.
    
#     Outputs:
#     - cluster_centroids (Dict[str, Dict[int, np.ndarray]]): Mapping with method
#                     names as keys and dictionaries as values. The inner
#                     dictionaries have the number of clusters as keys and numpy
#                     arrays of the cluster centroids as values.
#     """
#     # for each method, for each number of clusters, get the clusters and compute their centroids
#     cluster_centroids = {}
    
#     # for each method, attain the mapping from # of clusters to labels
#     for method, nclust_to_labels in method_to_labels.items():
#         # store the centroids for each # of clusters
#         num_cluster_centroids = {}
#         for num_clusters, cluster_labels in nclust_to_labels.items():
#             # the centroids are the mean of the values in each cluster,
#             # which is a px1 vector
#             centroids = np.zeros((num_clusters, X.shape[1]))
#             # for each cluster 1, ..., c, get the indices and compute the mean
#             for cluster_num in range(num_clusters):
#                 cluster_indices = np.where(cluster_labels == cluster_num)[0]
#                 cluster_values = lfi_train_values[method][cluster_indices]
#                 centroids[cluster_num] = np.mean(cluster_values, axis = 0)
#             num_cluster_centroids[num_clusters] = centroids
#         cluster_centroids[method] = num_cluster_centroids
#     return cluster_centroids

def get_cluster_centroids(lfi_train_values: Dict[str, np.ndarray],
                method_to_indices: Dict[str, Dict[int, Dict[int, np.ndarray]]]):
    """
    Gets the middle of each cluster so that test data can be assigned.
    
    Inputs:
    - lfi_train_values (Dict[str, np.ndarray]): The LMDI+ values for the
                    training data. The keys correspond to the method names and
                    the values are numpy arrays of the LMDI+ values.
    - method_to_indices (Dict[str, Dict[int, Dict[int, np.ndarray]]]): Mapping
                    with method names as keys and dictionaries as values. The
                    inner dictionaries have the number of clusters as keys and
                    dictionaries as values. The innermost dictionaries have the
                    cluster numbers as keys and numpy arrays of the indices in
                    each cluster as values.
    
    Outputs:
    - cluster_centroids (Dict[str, Dict[int, np.ndarray]]): Mapping with method
                    names as keys and dictionaries as values. The inner
                    dictionaries have the number of clusters as keys and numpy
                    arrays of the cluster centroids as values.
    """
    
    # map to be built out and returned
    cluster_centroids = {}
    
    # nclust_map has # of clusters as keys and an interior mapping of the
    # clusters for that # of clusters as vals
    for variant, nclust_map in method_to_indices.items():
        # store the centroids for each # of clusters
        nclust_centroids = {}
        # cluster_map has cluster # as keys and an array of indices as vals
        for nclust, cluster_map in nclust_map.items():
            # centroids is a matrix of shape (c, p) where c is the number of
            # clusters and p is the number of features
            centroids = np.zeros((nclust, lfi_train_values[variant].shape[1]))
            for c, idxs in cluster_map.items():
                # assign to row c the mean of the values in the cluster
                centroids[c] = np.mean(lfi_train_values[variant][idxs], axis=0)
            # store the centroids for this number of clusters
            nclust_centroids[nclust] = centroids
        # store the centroids for this variant
        cluster_centroids[variant] = nclust_centroids
    
    return cluster_centroids

def get_test_clusters(lfi_test_values: Dict[str, np.ndarray],
                      cluster_centroids: Dict[str, Dict[int, np.ndarray]]):
    """
    Assign the test observations to the closest centroid.
    
    Inputs:
    - lfi_test_values (Dict[str, np.ndarray]): The LMDI+ values for the
                    testing data. The keys correspond to the method names and
                    the values are numpy arrays of the LMDI+ values.
    - cluster_centroids (Dict[str, Dict[int, np.ndarray]]): Mapping with method
                    names as keys and dictionaries as values. The inner
                    dictionaries have the number of clusters as keys and numpy
                    arrays of the cluster centroids as values.
    
    Outputs:
    - test_clusters (Dict[str, Dict[int, Dict[int, np.ndarray]]]): Mapping with
                    method names as keys and dictionaries as values. The inner
                    dictionaries have the number of clusters as keys and
                    dictionaries as values. The innermost dictionaries have the
                    cluster numbers as keys and numpy arrays of the indices in
                    each cluster as values.
    """

    # we first attain a map from variant to cluster assignments
    method_to_labels = {}
    
    # for each variant, there is a map from the number of clusters to the
    # cluster centroid array
    for variant, centroid_map in cluster_centroids.items():
        # each variant will have a mapping to store its clusters.
        # the number of clusters will be the key, and the array of cluster
        # assignments will be the value.
        num_cluster_map = {}
        # for each # of clusters, get the centroids and see which is closest
        for nclust, centroid_array in centroid_map.items():
            # store the cluster assignments for each test observation
            cluster_membership = np.zeros(len(lfi_test_values[variant]))
            # for each test observation, find the closest centroid
            for i, test_value in enumerate(lfi_test_values[variant]):
                # subtracting (p,) from (c, p) broadcasts to (c, p)
                distances = np.linalg.norm(centroid_array - test_value, axis=1)
                # assign to the closest centroid
                cluster_membership[i] = np.argmin(distances)
            # store the cluster assignments for this number of clusters
            num_cluster_map[nclust] = cluster_membership
        # store the mapping for the variant
        method_to_labels[variant] = num_cluster_map
        
    # convert the cluster assignments to indices
    method_to_indices = {}
    for variant, nclust_map in method_to_labels.items():
        # this will be a mapping from the number of clusters to another mapping,
        # which will be from the cluster number to the indices in that cluster.
        num_cluster_map = {}
        for num_clusters, cluster_labels in nclust_map.items():
            # this is the inner mapping mentioned above
            cluster_map = {}
            for cluster_num in range(num_clusters):
                # for each cluster, get the indices with that cluster label
                cluster_indices = np.where(cluster_labels == cluster_num)[0]
                cluster_map[cluster_num] = cluster_indices
            num_cluster_map[num_clusters] = cluster_map
        method_to_indices[variant] = num_cluster_map
    
    return method_to_indices

def compute_performance(X_train: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_test: np.ndarray,
                    train_clusters: Dict[str, Dict[int, Dict[int, np.ndarray]]],
                    test_clusters: Dict[str, Dict[int, Dict[int, np.ndarray]]],
                    task: str):
    """
    Fit regression models on the train data for each cluster, and calculate
    the performance on the test data.
    
    Inputs:
    - X_train (np.ndarray): The feature matrix for the training set.
    - X_test (np.ndarray): The feature matrix for the testing set.
    - y_train (np.ndarray): The target vector for the training set.
    - y_test (np.ndarray): The target vector for the testing set.
    - train_clusters (Dict[str, Dict[int, Dict[int, np.ndarray]]]): Training
                    cluster representation.
    - test_clusters (Dict[str, Dict[int, Dict[int, np.ndarray]]]): Testing
                    cluster representation.
    - task (str): The task type, either 'classification' or 'regression'.
    
    Outputs:
    - metrics_to_variants (Dict[str, Dict[str, Dict[int, float]]]): Mapping from
                metrics (str) -> variants (str) -> nclust (int) -> score (float)
    """
    
    # create a mapping of metrics to measure
    if task == "classification":
        # metrics = {"accuracy": accuracy_score, "roc_auc": roc_auc_score,
        #            "average_precision": average_precision_score,
        #            "f1": f1_score, "log_loss": log_loss}
        metrics = {"accuracy": accuracy_score} # forget others since they aren't
        # defined if only one class is present
    else:
        metrics = {"r2": r2_score, "rmse": root_mean_squared_error}
    
    # metrics (str) -> variants (str) -> nclust (int) -> score (float)
    metrics_to_variants = {}
    for metric_name, metric_func in metrics.items():
        variants_to_nclust = {}
        for variant, nclust_map in train_clusters.items():
            nclust_to_score = {}
            # for each number of clusters, get each cluster, fit a model, and
            # calculate the metric
            for nclust in range(2, 11):
                # store scores in list in case some clusters have no test points
                cluster_scores = []
                cluster_sizes = []
                # c = 1, ..., nclust, get the cluster and fit a model
                for c in range(nclust):
                    # for train we can use nclust_map, but for test
                    # we need to use test_clusters, since nclust_map is the
                    # value for the training data
                    X_cluster_train = X_train[nclust_map[nclust][c]]
                    y_cluster_train = y_train[nclust_map[nclust][c]]
                    X_cluster_test = X_test[test_clusters[variant][nclust][c]]
                    y_cluster_test = y_test[test_clusters[variant][nclust][c]]
                    
                    # if no test points have been assigned to this cluster, skip
                    if X_cluster_test.shape[0] == 0:
                        continue
                    
                    # fit regression model to the cluster's training data
                    if task == "classification":
                        # check if the train cluster has only one class
                        if len(np.unique(y_cluster_train)) == 1:
                            print(f"For {nclust} clusters, cluster #{c} in variant {variant} has only " + \
                                  "one class. Predicting that class for all " + \
                                      "test points.")
                            # if so, predict that class for all test points
                            y_cluster_pred = np.ones(X_cluster_test.shape[0])* \
                                            y_cluster_train[0]
                            cluster_scores.append(metric_func(y_cluster_test,
                                                              y_cluster_pred))
                            cluster_sizes.append(X_cluster_test.shape[0])
                            continue
                        model = LogisticRegression()
                    else:
                        model = LinearRegression()
                    model.fit(X_cluster_train, y_cluster_train)
                    
                    # store the cluster scores and sizes for weighted average
                    y_cluster_pred = model.predict(X_cluster_test)
                    if task == "regression" and metric_name == "rmse" and variant == "lasso_l2_signed_nonnormed_noleafavg_rank" and nclust == 9:
                        print(f"Cluster {c} in variant {variant} has RMSE {metric_func(y_cluster_test, y_cluster_pred)}")
                        # print coefficents of the model
                        print("model coef:")
                        print(model.coef_)
                    cluster_scores.append(metric_func(y_cluster_test,
                                                      y_cluster_pred))
                    cluster_sizes.append(X_cluster_test.shape[0])
                
                # now back in loop over nclust
                nclust_to_score[nclust] = \
                                    weighted_metric(np.array(cluster_scores),
                                                    np.array(cluster_sizes))
            # now back in loop over variants
            variants_to_nclust[variant] = nclust_to_score
        # now back in loop over metrics
        metrics_to_variants[metric_name] = variants_to_nclust

    return metrics_to_variants

def write_results(result_dir: str, dataid: int, seed: int, clustertype: str,
                  task: str,
                  metrics_to_scores: Dict[str, Dict[str, Dict[int, float]]]):
    """
    Writes the results to a csv file.
    
    Inputs:
    - result_dir (str): The directory to save the results.
    - dataid (int): The OpenML dataset ID.
    - seed (int): The random seed used.
    - clustertype (str): The clustering method used.
    - task (str): The task type, either 'classification' or 'regression'.
    - metrics_to_scores (Dict[str, Dict[str, Dict[int, float]]]): Results
                                            calculated from compute_performance.
    
    Outputs:
    - None
    """
    
    # for each metric, save the results
    for metric_name in metrics_to_scores.keys():
        # write the results to a csv file
        print(f"Saving {metric_name} results...")
        for variant in metrics_to_scores[metric_name].keys():
            # create dataframe with # of clusters and scores as columns
            df = pd.DataFrame(
                list(metrics_to_scores[metric_name][variant].items()),
                columns=["nclust", f"{metric_name}"]
                )
            # if the path does not exist, create it
            if not os.path.exists(oj(result_dir, f"{task}/dataid{dataid}/seed{seed}"+ \
                                     f"/{metric_name}/{clustertype}")):
                os.makedirs(oj(result_dir, f"{task}/dataid{dataid}/seed{seed}" + \
                               f"/{metric_name}/{clustertype}"))
            # save the dataframe to a csv file
            df.to_csv(oj(result_dir, f"{task}/dataid{dataid}/seed{seed}/" + \
                         f"{metric_name}/{clustertype}", f"{variant}.csv"))

    return

def run_pipeline1(seed: int, dataid: int, clustertype: str):
    """
    Run pipeline 1 for the subgroup experiments.
    
    Inputs:
    - seed (int): The random seed to use.
    - dataid (int): The OpenML dataset ID.
    - clustertype (str): The clustering method to use.
    
    Outputs:
    - None
    """
    
    # get data
    X, y = get_openml_data(dataid)
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                        random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                        test_size = 0.5,
                                                        random_state=seed)
    
    # check if task is regression or classification
    if len(np.unique(y)) == 2:
        task = 'classification'
    else:
        task = 'regression'
        
    # fit the prediction models
    rf, rf_plus_baseline, rf_plus = fit_models(X_train, y_train, task)
    
    rf_plus_ridge = RandomForestPlusRegressor(rf_model=rf,
                                              prediction_model=RidgeCV(cv=5))
    rf_plus_ridge.fit(X_train, y_train)
    
    rf_plus_lasso = RandomForestPlusRegressor(rf_model=rf,
                                              prediction_model=LassoCV(cv=5,
                                                max_iter=10000, random_state=0))
    rf_plus_lasso.fit(X_train, y_train)
    
    # obtain shap feature importances
    shap_explainer = shap.TreeExplainer(rf)
    shap_train_values, shap_train_rankings = get_shap(X_val, shap_explainer,
                                                      task)
    shap_test_values, shap_test_rankings = get_shap(X_test, shap_explainer,
                                                    task)
    
    # get lime feature importances
    lime_train_values, lime_train_rankings = get_lime(X_val, rf, task)
    lime_test_values, lime_test_rankings = get_lime(X_test, rf, task)
    
    # create list of lmdi variants
    lmdi_variants = create_lmdi_variant_map()
    
    # obtain lmdi feature importances
    lmdi_explainers = get_lmdi_explainers(rf_plus, lmdi_variants,
                                          rf_plus_baseline = rf_plus_baseline,
                                          rf_plus_lasso = rf_plus_lasso,
                                          rf_plus_ridge = rf_plus_ridge)
    lfi_train_values, lfi_train_rankings = get_lmdi(X_train, y_train,
                                                    lmdi_variants,
                                                    lmdi_explainers)
    lfi_train_values, lfi_train_rankings = get_lmdi(X_val, None,
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
    
    # add the raw data to the dictionaries as a baseline of comparison
    lfi_train_values["rawdata"] = X_val
    lfi_test_values["rawdata"] = X_test
    
    # add lime to the dictionaries
    lfi_train_values["lime"] = lime_train_values
    lfi_test_values["lime"] = lime_test_values
        
    # get the clusterings
    # method_to_labels, method_to_indices = get_train_clusters(lfi_train_values, clustertype)
    train_clusters = get_train_clusters(lfi_train_values, clustertype)
    cluster_centroids = get_cluster_centroids(lfi_train_values, train_clusters)
    test_clusters = get_test_clusters(lfi_test_values, cluster_centroids)
    
    # compute the performance
    metrics_to_scores = compute_performance(X_val, X_test, y_val, y_test,
                                            train_clusters, test_clusters, task)
    
    # save the results
    result_dir = oj(os.path.dirname(os.path.realpath(__file__)),
                    'results/pipeline1/')
    write_results(result_dir, dataid, seed, clustertype, metrics_to_scores)
    
    print("Results saved!")
    
    return

def run_pipeline2(seed: int, dataid: int, clustertype: str, standarize: bool):
    """
    Run pipeline 2 for the subgroup experiments.
    
    Inputs:
    - seed (int): The random seed to use.
    - dataid (int): The OpenML dataset ID.
    - clustertype (str): The clustering method to use.
    - standardize (bool): Whether to standardize the data.
    
    Outputs:
    - None
    """
    
    # get data
    X, y = get_openml_data(dataid, standardize)
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5,
                                                        random_state=seed)
    
    # check if task is regression or classification
    if len(np.unique(y)) == 2:
        task = 'classification'
    else:
        task = 'regression'
        
    # fit the prediction models
    rf, rf_plus_baseline, rf_plus_ridge, rf_plus_lasso, rf_plus_elastic = \
        fit_models(X_train, y_train, task)
    
    # obtain shap feature importances
    shap_explainer = shap.TreeExplainer(rf)
    shap_test_values, shap_test_rankings = get_shap(X_test, shap_explainer,
                                                    task)
    
    # get lime feature importances
    lime_test_values, lime_test_rankings = get_lime(X_test, rf, task)
    
    # create list of lmdi variants
    lmdi_variants = create_lmdi_variant_map()
    
    # obtain lmdi feature importances
    # lmdi_explainers = get_lmdi_explainers(rf_plus, lmdi_variants,
    #                                       rf_plus_baseline = rf_plus_baseline,
    #                                       rf_plus_lasso = rf_plus_lasso,
    #                                       rf_plus_ridge = rf_plus_ridge)
    lmdi_explainers = get_lmdi_explainers(rf_plus_baseline, rf_plus_ridge,
                                          rf_plus_lasso, rf_plus_elastic,
                                          lmdi_variants)
    # we don't actually want to use the training values, but for leaf averaging
    # variants, we need to have the training data to compute the leaf averages
    lfi_train_values, lfi_train_rankings = get_lmdi(X_train, y_train,
                                                    lmdi_variants,
                                                    lmdi_explainers)
    lfi_test_values, lfi_test_rankings = get_lmdi(X_test, None,
                                                  lmdi_variants,
                                                  lmdi_explainers)
    # add shap to the dictionaries
    lfi_test_values["shap"] = shap_test_values
    lfi_test_rankings["shap"] = shap_test_rankings
    
    # add the raw data to the dictionaries as a baseline of comparison
    lfi_test_values["rawdata"] = X_test
    
    # add lime to the dictionaries
    lfi_test_values["lime"] = lime_test_values
        
    # get the clusterings - while we are not doing this on the training values,
    # the get_train_clusters function still does what we want it to.
    clusters = get_train_clusters(lfi_test_values, clustertype)
    
    # for each cluster, assign half of the indices to the "fitting" set and
    # the other half to the "evaluation" set
    fitting_clusters = {}
    evaluation_clusters = {}
    for variant, nclust_map in clusters.items():
        fitting_nclust_to_c = {}
        evaluation_nclust_to_c = {}
        for nclust, cluster_map in nclust_map.items():
            fitting_c_to_idxs = {}
            evaluation_c_to_idxs = {}
            for c, idxs in cluster_map.items():
                if len(idxs) < 3:
                    # warning message that the cluster is too small
                    warnings.warn(f"For {nclust} clusters, cluster #{c} in " + \
                        f"variant {variant} has fewer than 3 observations.",
                        Warning)
                # shuffle the indices and split them in half
                np.random.shuffle(idxs)
                half = len(idxs) // 2
                # NOTE: half: and :half were switched below.
                fitting_c_to_idxs[c] = idxs[half:]
                evaluation_c_to_idxs[c] = idxs[:half]
            fitting_nclust_to_c[nclust] = fitting_c_to_idxs
            evaluation_nclust_to_c[nclust] = evaluation_c_to_idxs
        fitting_clusters[variant] = fitting_nclust_to_c
        evaluation_clusters[variant] = evaluation_nclust_to_c
        
    # compute the performance - we are using test data for both, not an error
    metrics_to_scores = compute_performance(X_test, X_test, y_test, y_test,
                                            fitting_clusters,
                                            evaluation_clusters, task)
    
    # save the results
    # result_dir = oj(os.path.dirname(os.path.realpath(__file__)),
    #                 'results/pipeline2/')
    result_dir = oj(os.path.dirname(os.path.realpath(__file__)),
                    'results/')
    write_results(result_dir, dataid, seed, clustertype, task,
                  metrics_to_scores)
    
    print("Results saved!")
    
    return
    
        
if __name__ == '__main__':
    
    # store command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dataid', type=int, default=None)
    parser.add_argument('--pipeline', type=int, default=None)
    parser.add_argument('--clustertype', type=str, default=None)
    parser.add_argument('--standardize', type=int, default=None)
    args = parser.parse_args()
    
    # convert namespace to a dictionary
    args_dict = vars(args)

    # assign the arguments to variables
    seed = args_dict['seed']
    dataid = args_dict['dataid']
    pipeline = args_dict['pipeline']
    clustertype = args_dict['clustertype']
    standardize = args_dict['standardize']
    
    # enforce that standardize is either 0 or 1
    if standardize not in [0, 1]:
        raise ValueError("Standardize must be 0 or 1.")
    if standardize == 1:
        standardize = True
    else:
        standardize = False
    
    # enforce that pipeline either needs to in [1, 2]
    if pipeline not in [1, 2]:
        raise ValueError("Pipeline must be 1 or 2.")
    
    ### PIPELINE 1 ###
    
    if pipeline == 1:
        print(f"Running pipeline 1 with data ID {dataid} ...")
        run_pipeline1(seed, dataid, clustertype)
    else:
        print(f"Running pipeline 2 with data ID {dataid}...")
        run_pipeline2(seed, dataid, clustertype, standardize)