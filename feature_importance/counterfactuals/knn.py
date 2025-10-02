# data science imports
import numpy as np

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
    
    # run pipeline
    raw_distances_rf, shap_distances_rf, lime_distances_rf, lmdi_distances_rf, lmdi_plus_distances_rf = \
        perform_rf_pipeline(k, data_id, nbr_dist, cfact_dist, use_preds)
    raw_distances_gb, shap_distances_gb, lime_distances_gb, lmdi_plus_distances_gb = \
        perform_gb_pipeline(k, data_id, nbr_dist, cfact_dist, use_preds)

    # save results
    use_preds_str = "preds" if use_preds else "oracle"
    for method in ["raw", "shap", "lime", "lmdi", "lmdi_plus"]:
        make_rf_dir = oj("results", "rf", f"{data_source}_{data_id}", method, use_preds_str, f"k{k}")
        make_gb_dir = oj("results", "gb", f"{data_source}_{data_id}", method, use_preds_str, f"k{k}")
        os.makedirs(make_rf_dir, exist_ok=True)
        os.makedirs(make_gb_dir, exist_ok=True)
    print(f"Saving results")

    np.savetxt(oj("results", "rf", f"{data_source}_{data_id}", "raw", use_preds_str, f"k{k}", f"nbr-dist-{nbr_dist}_cfact-dist-{cfact_dist}.csv"), raw_distances_rf, delimiter=",")
    np.savetxt(oj("results", "rf", f"{data_source}_{data_id}", "shap", use_preds_str, f"k{k}", f"nbr-dist-{nbr_dist}_cfact-dist-{cfact_dist}.csv"), shap_distances_rf, delimiter=",")
    np.savetxt(oj("results", "rf", f"{data_source}_{data_id}", "lime", use_preds_str, f"k{k}", f"nbr-dist-{nbr_dist}_cfact-dist-{cfact_dist}.csv"), lime_distances_rf, delimiter=",")
    np.savetxt(oj("results", "rf", f"{data_source}_{data_id}", "lmdi", use_preds_str, f"k{k}", f"nbr-dist-{nbr_dist}_cfact-dist-{cfact_dist}.csv"), lmdi_distances_rf, delimiter=",")
    np.savetxt(oj("results", "rf", f"{data_source}_{data_id}", "lmdi_plus", use_preds_str, f"k{k}", f"nbr-dist-{nbr_dist}_cfact-dist-{cfact_dist}.csv"), lmdi_plus_distances_rf, delimiter=",")

    np.savetxt(oj("results", "gb", f"{data_source}_{data_id}", "raw", use_preds_str, f"k{k}", f"nbr-dist-{nbr_dist}_cfact-dist-{cfact_dist}.csv"), raw_distances_gb, delimiter=",")
    np.savetxt(oj("results", "gb", f"{data_source}_{data_id}", "shap", use_preds_str, f"k{k}", f"nbr-dist-{nbr_dist}_cfact-dist-{cfact_dist}.csv"), shap_distances_gb, delimiter=",")
    np.savetxt(oj("results", "gb", f"{data_source}_{data_id}", "lime", use_preds_str, f"k{k}", f"nbr-dist-{nbr_dist}_cfact-dist-{cfact_dist}.csv"), lime_distances_gb, delimiter=",")
    np.savetxt(oj("results", "gb", f"{data_source}_{data_id}", "lmdi_plus", use_preds_str, f"k{k}", f"nbr-dist-{nbr_dist}_cfact-dist-{cfact_dist}.csv"), lmdi_plus_distances_gb, delimiter=",")