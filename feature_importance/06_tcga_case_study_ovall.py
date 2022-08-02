import copy
import os.path
import pickle
import argparse

import pandas as pd
import numpy as np
from os.path import join as oj
from tqdm import tqdm

import sys
sys.path.append("..")

from imodels.importance import R2FExp, GeneralizedMDI, GeneralizedMDIJoint
from imodels.importance import LassoScorer, RidgeScorer, ElasticNetScorer, RobustScorer, LogisticScorer, JointRidgeScorer, JointLogisticScorer, JointRobustScorer, JointALOLogisticScorer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import shap
from feature_importance.scripts.mda import MDA
import sklearn.metrics as metrics


def multiclass_f1_score(y_onehot, ypreds, sample_weight=None):
    ypreds_label = ypreds.argmax(axis=1)
    y_label = y_onehot.argmax(axis=1)
    results = np.zeros(ypreds.shape[1])
    for k in range(ypreds.shape[1]):
        ypreds_k = (ypreds_label == k).astype(int)
        y_k = (y_label == k).astype(int)
        results[k] = metrics.f1_score(y_k, ypreds_k)
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', type=int, default=0)
    args = parser.parse_args()
    
    rep = args.rep
    
    # input parameters
    # N_REPS = 10
    # FI_MODELS = ["gjmdi_ridge_loocv", "gjmdi_logistic", "gjmdi_logistic_loocv", "mdi", "shap", "permutation", "mda"]
    FI_MODELS = ["gjmdi_ridge_loocv", "gjmdi_logistic_logloss", "gjmdi_logistic_auprc", "gjmdi_logistic_auroc", "gjmdi_logistic_loocv_logloss", "gjmdi_logistic_loocv_auprc", "gjmdi_logistic_loocv_auroc"]
    SCRATCH_DIR = "/global/scratch/users/tiffanytang/feature_importance"

    # load data
    DATA_DIR = oj(SCRATCH_DIR, "data")
    X_df = pd.read_csv(oj(DATA_DIR, "X_tcga_var_filtered_log_transformed.csv"))
    y_multiclass = pd.read_csv(oj(DATA_DIR, "Y_tcga.csv")).to_numpy().ravel()
    keep_idx = y_multiclass != "Normal"
    y_multiclass = y_multiclass[keep_idx]
    X_df = X_df[keep_idx]
    X = X_df.to_numpy()
    X = (X - X.mean()) / X.std()

    for c in np.unique(y_multiclass):
        y = np.ones(y_multiclass.shape)
        y[y_multiclass != c] = -1
        
        OUTPUT_DIR = oj(SCRATCH_DIR, "results", "tcga_brca_normalized", c, "rep{}".format(args.rep))
        # OUTPUT_DIR = oj(SCRATCH_DIR, "results", "tcga_brca", c, "rep{}".format(args.rep))
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # initialize outputs
        imp_values_dict = {}
    
        for fi_model in FI_MODELS:
            # rf_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, max_features="sqrt", random_state=rep)
            rf_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, max_features="sqrt", random_state=rep)
    
            if fi_model == "gjmdi_ridge_loocv":
                loocv_scorer = JointRidgeScorer(criterion="gcv", metric="loocv")
                gjMDI_loocv = GeneralizedMDIJoint(copy.deepcopy(rf_model), scorer=loocv_scorer, normalize_raw=True, oob=False, random_state=331)
                imp_values = gjMDI_loocv.get_importance_scores(X, y)#, diagnostics=True)
                                                                   
            if fi_model == "gjmdi_logistic_loocv_logloss":
                loocv_scorer = JointALOLogisticScorer(metric=metrics.log_loss)
                gjMDI_loocv = GeneralizedMDIJoint(copy.deepcopy(rf_model), scorer=loocv_scorer, normalize_raw=True, oob=False, random_state=331)
                imp_values = gjMDI_loocv.get_importance_scores(X, y)#, diagnostics=True)
                
            if fi_model == "gjmdi_logistic_loocv_auprc":
                loocv_scorer = JointALOLogisticScorer(metric=metrics.average_precision_score)
                gjMDI_loocv = GeneralizedMDIJoint(copy.deepcopy(rf_model), scorer=loocv_scorer, normalize_raw=True, oob=False, random_state=331)
                imp_values = gjMDI_loocv.get_importance_scores(X, y)#, diagnostics=True)
                
            if fi_model == "gjmdi_logistic_loocv_auroc":
                loocv_scorer = JointALOLogisticScorer(metric=metrics.roc_auc_score)
                gjMDI_loocv = GeneralizedMDIJoint(copy.deepcopy(rf_model), scorer=loocv_scorer, normalize_raw=True, oob=False, random_state=331)
                imp_values = gjMDI_loocv.get_importance_scores(X, y)#, diagnostics=True)
                
            if fi_model == "gjmdi_logistic_logloss":
                scorer = JointLogisticScorer(metric=metrics.log_loss)
                gjMDI = GeneralizedMDIJoint(copy.deepcopy(rf_model), scorer=scorer, normalize_raw=True, oob=False, random_state=331)
                imp_values = gjMDI.get_importance_scores(X, y)#, diagnostics=True)
                
            if fi_model == "gjmdi_logistic_auprc":
                scorer = JointLogisticScorer(metric=metrics.average_precision_score)
                gjMDI = GeneralizedMDIJoint(copy.deepcopy(rf_model), scorer=scorer, normalize_raw=True, oob=False, random_state=331)
                imp_values = gjMDI.get_importance_scores(X, y)#, diagnostics=True)
                
            if fi_model == "gjmdi_logistic_auroc":
                scorer = JointLogisticScorer(metric=metrics.roc_auc_score)
                gjMDI = GeneralizedMDIJoint(copy.deepcopy(rf_model), scorer=scorer, normalize_raw=True, oob=False, random_state=331)
                imp_values = gjMDI.get_importance_scores(X, y)#, diagnostics=True)
    
            if fi_model == "gjmdi_f1":
                f1_scorer = JointRidgeScorer(criterion="gcv", metric=multiclass_f1_score)
                gjMDI_f1 = GeneralizedMDIJoint(copy.deepcopy(rf_model), scorer=f1_scorer, normalize_raw=True, oob=False, random_state=331)
                imp_values = gjMDI_f1.get_importance_scores(X, y)#, diagnostics=True)
    
            rf_model.fit(X, y)
    
            if fi_model == "mdi":
                imp_values = rf_model.feature_importances_
    
            if fi_model == "permutation":
                rf_model_perm = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, max_features="sqrt", random_state=rep)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=rep + 500)
                rf_model_perm.fit(X_train, y_train)
                perm_fit = permutation_importance(rf_model_perm, X_test, y_test, n_repeats=5, random_state=rep + 100)
                imp_values = perm_fit.importances_mean
    
            if fi_model == "shap":
                explainer = shap.TreeExplainer(rf_model)
                shap_values = explainer.shap_values(X)
                imp_values = np.abs(shap_values).mean(axis=0)
    
            if fi_model == "mda":
                y_factorized = pd.factorize(y.ravel())
                y_mda = y_factorized[0]
                rf_model_mda = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, max_features="sqrt", random_state=rep)
                rf_model_mda.fit(X, y_mda)
                imp_values, _ = MDA(rf_model_mda, X, y_mda[:, np.newaxis], type="oob", n_trials=5, metric="accuracy")
                
            imp_values_dict[fi_model] = copy.deepcopy(imp_values)
            pickle.dump(imp_values_dict, open(oj(OUTPUT_DIR, "imp_values_dict.pickle"), "wb"))
    
#%%
