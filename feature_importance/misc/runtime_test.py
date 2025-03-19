import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusRegressor
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import RFPlusMDI
import argparse
import os
from os.path import join as oj

def run_experiment(n_samples, n_features, lfi_method):
    
    # X = np.load("X.npy")
    # y = np.load("y.npy")
    
    # create dataframe to store time results
    time_results = pd.DataFrame(columns=['n_samples', 'n_features', 'method',
        'check_data_time', 'fit_rf_time', 'fit_forest_time',
        'init_transformer_time', 'get_transformed_data_time', 'fit_prediction_model_time',
        'average_tree_time', 'init_ppm_time', 'get_leafs_in_test_samples_time',
        'partial_predictions_time', 'mean_partial_pred_time_per_estimator',
        'mean_partial_pred_k_time_per_estimator', 'leaf_average_time',
        'get_lfi_time'])
    time_row = []
    
    # put n_samples and n_features in time_results
    time_row.append(n_samples)
    time_row.append(n_features)
    time_row.append(lfi_method)
    
    # generate normally distributed data with n_samples and n_features
    X = np.random.normal(size=(n_samples, n_features))
    # make y the sum of half of the features plus noise
    y = np.sum(X[:, :n_features//2], axis=1) + np.random.normal(size=n_samples)
    
    # initialize RF model
    rf = RandomForestRegressor(n_estimators = 100, max_depth = 5, min_samples_leaf = 5, max_features = 0.33, random_state = 42)

    # fit RF+ model
    rf_plus = RandomForestPlusRegressor(rf_model=rf, prediction_model=SGDRegressor(alpha = 0.001))
    rf_plus.fit(X, y)
    
    # get feature importance
    rf_plus_explainer = RFPlusMDI(rf_plus)
    if lfi_method == "linear_partial":
        lmdi = rf_plus_explainer.explain_linear_partial(X, y, leaf_average = True, l2norm=True, njobs = -1)
    else:
        lmdi = rf_plus_explainer.explain_r2(X, y, l2norm=True)
        
    time_row.append(rf_plus.check_data_time)
    time_row.append(rf_plus.fit_rf_time)
    time_row.append(rf_plus.fit_forest_time)
    time_row = time_row + list(rf_plus.fit_trees_time.mean())
    time_row.append(rf_plus_explainer.init_ppm_time)
    time_row.append(rf_plus_explainer.get_leafs_in_test_samples_time)
    time_row.append(rf_plus_explainer.partial_predictions_time)
    lst = list()
    lst2 = list()
    for explainer in rf_plus_explainer.tree_explainers:
        lst.append(explainer._total_partial_preds_time)
        lst2.append(explainer._partial_preds_time)
    time_row.append(np.mean(np.array(lst)))
    time_row.append(np.mean(np.array(lst2)))
    time_row.append(rf_plus_explainer.leaf_average_time)
    time_row.append(rf_plus_explainer.get_lfi_time)
    
    # append time_row to time_results
    time_results.loc[0] = time_row
    
    return time_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_dir = os.getenv("SCRATCH")
    if default_dir is not None:
        default_dir = oj(default_dir, "feature_importance", "results")
    else:
        default_dir = oj(os.path.dirname(os.path.realpath(__file__)), 'results')
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--n_features', type=int, default=None)
    parser.add_argument('--lfi_method', type=str, default=None)
    parser.add_argument('--rep', type=int, default=None)
    args = parser.parse_args()
    
    # Convert Namespace to a dictionary
    args_dict = vars(args)

    # Assign each key-value pair to a variable
    n_samples = args_dict['n_samples']
    n_features = args_dict['n_features']
    lfi_method = args_dict['lfi_method']
    rep = args_dict['rep']
    print("Running time experiment getting LFI using", lfi_method, "with", n_samples, "rows and", n_features, "features for the", rep, "th time.")
    time_results = run_experiment(n_samples, n_features, lfi_method)
    time_results.to_csv(oj(default_dir, f'new_time_results_n{n_samples}_p{n_features}_method_{lfi_method}_rep{rep}.csv'), index=False)
    print("Ran experiment successfully!")
    
    