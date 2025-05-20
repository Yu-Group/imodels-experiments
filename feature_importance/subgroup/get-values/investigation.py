# standard data science packages
import numpy as np

# functions for subgroup experiments
import shap

# sklearn imports
from sklearn.model_selection import train_test_split

# for saving results
import argparse
import os
from os.path import join as oj
import time

# subgroup imports
from subgroup import fit_models, create_lmdi_variant_map, get_lmdi_explainers, \
    get_lmdi, get_shap, get_lime

if __name__ == '__main__':
    
    # store command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--method', type=str, default=None)
    args = parser.parse_args()
    
    # convert namespace to a dictionary
    args_dict = vars(args)

    # assign the arguments to variables
    dataname = args_dict['dataname']
    seed = args_dict['seed']
    tree_method = args_dict['method']
    
    # check that tree_method is valid
    if tree_method != "rf":
        raise ValueError("Invalid tree method. Please choose 'rf'.")
    # if tree_method not in ["rf", "gb"]:
    #     raise ValueError("Invalid tree method. Please choose 'rf' or 'gb'.")
    
    print("Running Pipeline w/ " + dataname)

    dir_data = "../data_openml"
    
    X = np.loadtxt(oj(dir_data, f"X_{dataname}.csv"), delimiter=",")[1:,:]
    y = np.loadtxt(oj(dir_data, f"y_{dataname}.csv"), delimiter=",")[1:]
    
    # cast to np.float32
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    print("Step 1")
    
    starttime = time.time()

    # split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5,
                                                        random_state = seed)

    # fit random forest models
    rf, rf_plus_baseline, rf_plus_elastic = fit_models(X_train, y_train, "regression")
                
    endtime = time.time()

    print("Step 2: " + str(endtime - starttime) + " seconds")
    
    starttime = time.time()

    # create list of lmdi variants
    lmdi_variants = create_lmdi_variant_map()

    # obtain lmdi+ feature importances
    lmdi_explainers = get_lmdi_explainers(rf_plus_baseline, rf_plus_elastic,
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
    shap_explainer = shap.TreeExplainer(rf)
    shap_values, shap_rankings = get_shap(X_test, shap_explainer, "regression")
    
    endtime = time.time()

    print("Step 5: " + str(endtime - starttime) + " seconds")
    
    starttime = time.time()

    # obtain lime feature importances
    lime_values, lime_rankings = get_lime(X_test, rf, "regression")
        
    endtime = time.time()

    print("Step 6: " + str(endtime - starttime) + " seconds")
    
    # get the path to the parent directory of the current file
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    result_dir = oj(parent_dir, "lfi-values", f"seed{seed}")

    # if the path does not exist, create it
    if not os.path.exists(oj(result_dir, dataname)):
        os.makedirs(oj(result_dir, dataname))
        
    # print result directory
    print("Writing results to: " + oj(result_dir, dataname))

    # for each variant write the LFI values to a csv
    for variant in lfi_values.keys():
        np.savetxt(oj(result_dir, dataname, f"{variant}.csv"), lfi_values[variant], delimiter=",")
        
    np.savetxt(oj(result_dir, dataname, "shap.csv"), shap_values, delimiter=",")
    np.savetxt(oj(result_dir, dataname, "lime.csv"), lime_values, delimiter=",")
