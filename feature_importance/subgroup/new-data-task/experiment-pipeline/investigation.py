# standard data science packages
import numpy as np
import pandas as pd

# functions for subgroup experiments
import shap

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# data getter imports
# from data_loader import load_regr_data

# for saving results
import argparse
import os
from os.path import join as oj
import time

# subgroup imports
from subgroup import fit_gb_models, fit_models, create_lmdi_variant_map, get_lmdi_explainers, \
    get_lmdi, get_shap, get_lime

if __name__ == '__main__':
    
    # store command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--gender', type=int, default=False)
    args = parser.parse_args()
    
    # convert namespace to a dictionary
    args_dict = vars(args)

    # assign the arguments to variables
    dataname = args_dict['dataname']
    seed = args_dict['seed']
    tree_method = args_dict['method']
    gender = args_dict['gender']
    gender = bool(gender)
    print("Gender =", gender)
    
    # check that tree_method is valid
    if tree_method not in ["rf", "gb"]:
        raise ValueError("Invalid tree method. Please choose 'rf' or 'gb'.")
    
    print(f"Running Pipeline w/ {dataname}")

    dir_data = f"data/data_{dataname}"

    # X, y, names_covariates = load_regr_data(dataname, dir_data)
    
    # X = np.loadtxt(oj(dir_data, f"X.csv"), delimiter=",")[1:,:]
    # y = np.loadtxt(oj(dir_data, f"y.csv"), delimiter=",")[1:]
    X = pd.read_csv(oj(dir_data, "X.csv")).to_numpy()
    if not gender:
        if dataname == "parkinsons":
            # remove last col
            X = X[:, :-1]
        if dataname == "abalone":
            # remove first col
            X = X[:, 1:]
    y = pd.read_csv(oj(dir_data, "y.csv"), header = None).to_numpy().flatten()
    
    # cast to np.float32
    # X = X.astype(np.float32)
    # y = y.astype(np.float32)
    
    # standardize X with StandardScaler
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # standardize y
    # y = (y - np.mean(y)) / np.std(y)
    
    # if X has more than 5k rows, sample 5k rows of X and y
    # if X.shape[0] > 5000:
    #     np.random.seed(42)
    #     indices = np.random.choice(X.shape[0], 5000, replace=False)
    #     X = X[indices]
    #     y = y[indices]
    
    print("Step 1")
    
    starttime = time.time()

    # split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5,
                                                        random_state = seed)

    # if tree_method is "rf", fit random forest models
    if tree_method == "rf":
        rf, rf_plus_baseline, rf_plus_elastic = \
                fit_models(X_train, y_train, "regression")
    # if tree_method is "gb", fit gradient boosting models
    else:
        gb, gb_plus_baseline, gb_plus_ridge, gb_plus_lasso, gb_plus_elastic = \
                fit_gb_models(X_train, y_train, "regression")
                
    endtime = time.time()

    print("Step 2: " + str(endtime - starttime) + " seconds")
    
    starttime = time.time()

    # create list of lmdi variants
    lmdi_variants = create_lmdi_variant_map()

    # obtain lmdi feature importances
    if tree_method == "rf":
        lmdi_explainers = get_lmdi_explainers(rf_plus_baseline, rf_plus_elastic,
                                              lmdi_variants)
    else:
        lmdi_explainers = get_lmdi_explainers(gb_plus_baseline, gb_plus_ridge,
                                              gb_plus_lasso, gb_plus_elastic,
                                              lmdi_variants)
        
    endtime = time.time()
    
    print("Step 3: " + str(endtime - starttime) + " seconds")
    
    starttime = time.time()

    # we don't actually want to use the training values, but for leaf averaging
    # variants, we need to have the training data to compute the leaf averages
    lfi_values, lfi_rankings = get_lmdi(X_test, None, lmdi_variants,
                                        lmdi_explainers)
    
    endtime = time.time()

    print("Step 4: " + str(endtime - starttime) + " seconds")
    
    starttime = time.time()

    # obtain shap feature importances
    if tree_method == "rf":
        shap_explainer = shap.TreeExplainer(rf)
    else:
        shap_explainer = shap.TreeExplainer(gb)
    shap_values, shap_rankings = get_shap(X_test, shap_explainer, "regression")
    
    endtime = time.time()

    print("Step 5: " + str(endtime - starttime) + " seconds")
    
    starttime = time.time()

    # obtain lime feature importances
    if tree_method == "rf":
        lime_values, lime_rankings = get_lime(X_test, rf, "regression")
    else:
        lime_values, lime_rankings = get_lime(X_test, gb, "regression")
        
    endtime = time.time()

    print("Step 6: " + str(endtime - starttime) + " seconds")
    
    # get the path to the parent directory of the current file
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    result_dir = oj(parent_dir, "lfi-values", "fulldata", tree_method, f"seed{seed}")

    # if the path does not exist, create it
    if not gender:
        newdir = f"{dataname}-nogender"
    else:
        newdir = f"{dataname}"
    if not os.path.exists(oj(result_dir, newdir)):
        os.makedirs(oj(result_dir, newdir))

    # for each variant write the LFI values to a csv
    for variant in lfi_values.keys():
        np.savetxt(oj(result_dir, newdir, f"{variant}.csv"), lfi_values[variant], delimiter=",")
        
    np.savetxt(oj(result_dir, newdir, "shap.csv"), shap_values, delimiter=",")
    np.savetxt(oj(result_dir, newdir, "lime.csv"), lime_values, delimiter=",")
