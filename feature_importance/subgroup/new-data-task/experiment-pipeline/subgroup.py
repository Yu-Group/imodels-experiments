# standard data science packages
import numpy as np
import pandas as pd

# imodels imports
from imodels.tree.rf_plus.rf_plus.rf_plus_models import \
    RandomForestPlusRegressor, RandomForestPlusClassifier
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import RFPlusMDI

# functions for subgroup experiments
import shap
import lime

# sklearn imports
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV,\
    RidgeCV, ElasticNetCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
    GradientBoostingRegressor

# for function headers
from typing import Tuple, Dict

def fit_gb_models(X_train: np.ndarray, y_train: np.ndarray, task: str):
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

        # not supported, throw error
        raise ValueError("Gradient boosting not supported for classification.")
    
    # if regression, fit regressors
    elif task == "regression":
        
        # fit random forest with params from MDI+
        # gb = GradientBoostingRegressor(n_estimators=100, min_samples_leaf=5,
        #                                max_features=0.33, random_state=42)
        gb = GradientBoostingRegressor(random_state=42)
        gb.fit(X_train, y_train)
        
        # baseline rf+ includes no raw feature and only fits on in-bag samples
        gb_plus_baseline = RandomForestPlusRegressor(rf_model=gb,
                                        include_raw=False,
                                        prediction_model=LinearRegression())
        gb_plus_baseline.fit(X_train, y_train, n_jobs=None)
        
        # ridge rf+
        gb_plus_ridge = RandomForestPlusRegressor(rf_model=gb,
                                                prediction_model=RidgeCV(cv=5))
        gb_plus_ridge.fit(X_train, y_train, n_jobs=None)
        
        # lasso rf+
        gb_plus_lasso = RandomForestPlusRegressor(rf_model=gb,
                                                  prediction_model=LassoCV(cv=5,
                                            max_iter=10000, random_state=42))
        gb_plus_lasso.fit(X_train, y_train, n_jobs=None)
        
        # elastic net rf+
        gb_plus_elastic = RandomForestPlusRegressor(rf_model=gb,
                                            prediction_model=ElasticNetCV(cv=5,
                                        l1_ratio=[0.1,0.5,0.7,0.9,0.95,0.99],
                                        max_iter=10000,random_state=42))
        gb_plus_elastic.fit(X_train, y_train, n_jobs=None)
    
    # otherwise, throw error
    else:
        raise ValueError("Task must be 'classification' or 'regression'.")
    
    # return tuple of models
    return gb, gb_plus_baseline, gb_plus_ridge, gb_plus_lasso, gb_plus_elastic

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
        rf_plus_baseline.fit(X_train, y_train, n_jobs=None)
        
        # ridge rf+
        # rf_plus_ridge = RandomForestPlusClassifier(rf_model=rf,
        #                     prediction_model=LogisticRegressionCV(penalty='l2',
        #                                 cv=5, max_iter=10000, random_state=42))
        # rf_plus_ridge.fit(X_train, y_train)

        # lasso rf+
        # rf_plus_lasso = RandomForestPlusClassifier(rf_model=rf,
        #                     prediction_model=LogisticRegressionCV(penalty='l1',
        #                             solver = 'saga', cv=3, n_jobs=-1, tol=5e-4,
        #                             max_iter=5000, random_state=42))
        # rf_plus_lasso.fit(X_train, y_train)

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
        # rf_plus_ridge = RandomForestPlusRegressor(rf_model=rf,
        #                                         prediction_model=RidgeCV(cv=5))
        # rf_plus_ridge.fit(X_train, y_train)
        
        # lasso rf+
        # rf_plus_lasso = RandomForestPlusRegressor(rf_model=rf,
        #                                           prediction_model=LassoCV(cv=5,
        #                                     max_iter=10000, random_state=42))
        # rf_plus_lasso.fit(X_train, y_train)
        
        # elastic net rf+
        rf_plus_elastic = RandomForestPlusRegressor(rf_model=rf,
                                            prediction_model=ElasticNetCV(cv=5,
                                        l1_ratio=[0.1,0.5,0.7,0.9,0.95,0.99],
                                        max_iter=10000,random_state=42))
        rf_plus_elastic.fit(X_train, y_train)
    
    # otherwise, throw error
    else:
        raise ValueError("Task must be 'classification' or 'regression'.")
    
    # return tuple of models
    # return rf, rf_plus_baseline, rf_plus_ridge, rf_plus_lasso, rf_plus_elastic
    return rf, rf_plus_baseline, rf_plus_elastic

def create_lmdi_variant_map() -> Dict[str, Dict[str, bool]]:
    """
    Create a mapping of LMDI+ variants to argument mappings.
    
    Outputs:
    - lmdi_variants (Dict[str, Dict[str, bool]]): The LMDI variants to use.
    """
    
    # enumerate the different options when initializing a LMDI+ explainer.
    glm = ["elastic"]
    #l2norm = {True: "l2", False: "nonl2"}
    #sign = {True: "signed", False: "unsigned"}
    normalize = {True: "normed", False: "nonnormed"}
    square = {True: "squared", False: "nosquared"}
    leaf_average = {True: "leafavg", False: "noleafavg"}
    ranking = {True: "rank", False: "norank"}
    
    # create the mapping of variants to argument mappings
    lmdi_variants = {}
    for g in glm:
        for n in normalize:
            for s in square:
                for r in ranking:
                    if (not n) and (s):
                        continue
                    # create the name the variant will be stored under
                    variant_name = f"{g}_{normalize[n]}_{square[s]}_{ranking[r]}"
                    # store the arguments for the lmdi+ explainer
                    arg_map = {"glm": g, "normalize": n, "square": s,
                               "ranking": r}
                    lmdi_variants[variant_name] = arg_map
    # for g in glm:
    #     for l2 in l2norm:
    #         for s in sign:
    #             # sign is only relevant if l2norm is True
    #             if (not l2) and (s):
    #                 continue
    #             for n in normalize:
    #                 for la in leaf_average:
    #                     for r in ranking:
    #                         # ranking is only relevant if leaf_average is False
    #                         if la:
    #                             continue
    #                         # create the name the variant will be stored under
    #                         variant_name = f"{g}_{l2norm[l2]}_{sign[s]}" + \
    #                         f"_{normalize[n]}_{leaf_average[la]}_{ranking[r]}"
    #                         # store the arguments for the lmdi+ explainer
    #                         arg_map = {"glm": g, "l2norm": l2, "sign": s,
    #                                    "normalize": n, "leaf_average": la,
    #                                    "ranking": r}
    #                         lmdi_variants[variant_name] = arg_map
    
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

def get_lmdi_explainers(rf_plus_baseline, rf_plus_elastic,
                        lmdi_variants: Dict[str, Dict[str, bool]]):
    """
    Create the LMDI explainer objects for the subgroup experiments.
    
    Inputs:
    - rf_plus_baseline (RandomForestPlusClassifier/RandomForestPlusRegressor):
                    The baseline RF+ (no raw feature, only on in-bag samples,
                    regular linear/logistic prediction model).
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
        if lmdi_variants[variant_name]["glm"] == "elastic":
            lmdi_explainers[variant_name] = RFPlusMDI(rf_plus_elastic,
                                                      mode = "only_k",
                                                      evaluate_on = "all")
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
                                            normalize=False, square=False,
                                            ranking=False)

        # get the rankings using the method in the explainer class
        lmdi_rankings["lmdi_baseline"] = \
            lmdi_explainers["lmdi_baseline"].get_rankings(
                np.abs(lmdi_values["lmdi_baseline"])
                )
    
    # # if the explainer mapping has a lasso variant, we treat it differently
    # if "lmdi_lasso" in lmdi_explainers:
    #     # print("Computing LMDI+ for Lasso variant...")
    #     # lmdi_values["lmdi_lasso"] = \
    #     #     lmdi_explainers["lmdi_lasso"].explain_linear_partial(X, y,
    #     #                                     l2norm=True, sign=True,
    #     #                                     normalize=True, leaf_average=False,
    #     #                                     ranking=True)
    #     lmdi_values["lmdi_lasso"] = \
    #         lmdi_explainers["lmdi_lasso"].explain_linear_partial(X, y,
    #                                         l2norm=False, sign=False,
    #                                         normalize=False, leaf_average=False,
    #                                         ranking=True)
    #     # print("Done, values are:")
    #     # print(lmdi_values["lmdi_lasso"])
    
    # if "lmdi_ridge" in lmdi_explainers:
    #     # print("Computing LMDI+ for Ridge variant...")
    #     # lmdi_values["lmdi_ridge"] = \
    #     #     lmdi_explainers["lmdi_ridge"].explain_linear_partial(X, y,
    #     #                                     l2norm=True, sign=True,
    #     #                                     normalize=True, leaf_average=False,
    #     #                                     ranking=True)
    #     lmdi_values["lmdi_ridge"] = \
    #         lmdi_explainers["lmdi_ridge"].explain_linear_partial(X, y,
    #                                         l2norm=False, sign=False,
    #                                         normalize=False, leaf_average=False,
    #                                         ranking=True)
    #     # print("Done, values are:")
    #     # print
    
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
                                        normalize=variant_args["normalize"],
                                        square=variant_args["square"],
                                        ranking=variant_args["ranking"])
        
        # get rankings using the method in the explainer class. absolute value
        # taken to ensure that the rankings are based on the magnitude.
        lmdi_rankings[name] = explainer.get_rankings(np.abs(lmdi_values[name]))
        
    return lmdi_values, lmdi_rankings
