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

# local feature importance
import shap
import lime

# helpers
from knn_helper import *

# for saving results
import argparse
import os
from os.path import join as oj

if __name__ == "__main__":
    
    # store command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasource', type=str, default=None)
    parser.add_argument('--dataid', type=int, default=None)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--nbr_dist', type=str, default="l2")
    parser.add_argument('--cfact_dist', type=str, default="l2")
    parser.add_argument('--use_preds', type=int, default=0)
    args = parser.parse_args()
    
    # convert namespace to a dictionary
    args_dict = vars(args)

    # assign the arguments to variables
    data_source = args_dict['datasource']
    data_id = args_dict['dataid']
    k = args_dict['k']
    nbr_dist = args_dict['nbr_dist']
    cfact_dist = args_dict['cfact_dist']
    use_preds = args_dict['use_preds']
    if use_preds == 1:
        use_preds = True
    else:
        use_preds = False
    
    # check that each input is valid
    assert data_source in ["uci", "openml"], "data_source must be either 'uci' or 'openml'"
    assert data_id is not None, "data_id must be provided"
    assert k > 0, "k must be provided"
    assert nbr_dist in ["l1", "l2", "chebyshev"], "nbr_dist must be either 'l1', 'l2', or 'chebyshev'"
    assert cfact_dist in ["l1", "l2", "chebyshev"], "cfact_dist must be either 'l1', 'l2', or 'chebyshev'"
    
    # run pipeline
    shap_distances, lime_distances, lmdi_distances = perform_pipeline(k, data_id, nbr_dist, cfact_dist, use_preds)
    
    # save results
    # metrics = ["l1", "l2", "linfty"]
    use_preds_str = "preds" if use_preds else "oracle"
    results_dir = oj("results", f"{data_source}_{data_id}")
    for method in ["shap", "lime", "lmdi"]:
        make_dir = oj(results_dir, method, use_preds_str, f"k{k}")
        os.makedirs(make_dir, exist_ok=True)
    np.savetxt(oj(results_dir, "shap", use_preds_str, f"k{k}", f"nbr-dist-{nbr_dist}_cfact-dist-{cfact_dist}.csv"), shap_distances, delimiter=",")
    np.savetxt(oj(results_dir, "lime", use_preds_str, f"k{k}", f"nbr-dist-{nbr_dist}_cfact-dist-{cfact_dist}.csv"), lime_distances, delimiter=",")
    np.savetxt(oj(results_dir, "lmdi", use_preds_str, f"k{k}", f"nbr-dist-{nbr_dist}_cfact-dist-{cfact_dist}.csv"), lmdi_distances, delimiter=",")
    # for metric1 in metrics:
    #     for metric2 in metrics:
    #         np.savetxt(oj(results_dir, f"shap_distances_k{k}_{metric1}_{metric2}.csv"), shap_distances[metric1][metric2], delimiter=",")
    #         np.savetxt(oj(results_dir, f"lime_distances_k{k}_{metric1}_{metric2}.csv"), lime_distances[metric1][metric2], delimiter=",")
    #         np.savetxt(oj(results_dir, f"lmdi_distances_k{k}_{metric1}_{metric2}.csv"), lmdi_distances[metric1][metric2], delimiter=",")



