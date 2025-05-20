# standard data science packages
import numpy as np
import pandas as pd

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# hierarchical clustering imports
from scipy.cluster import hierarchy

# filesystem imports
import os
from os.path import join as oj

# for command-line args
import argparse

# other
from collections import defaultdict

dir_data = "../data_openml"

if __name__ == '__main__':
    
    # store command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--clustertype', type=str, default=None)
    parser.add_argument('--clustermodel', type=str, default=None)
    # parser.add_argument('--datafolder', type=str, default=None)
    # parser.add_argument('--methodname', type=str, default=None)
    args = parser.parse_args()
    
    # convert namespace to a dictionary
    args_dict = vars(args)

    # assign the arguments to variables
    dataname = args_dict['dataname']
    seed = args_dict['seed']
    clustertype = args_dict['clustertype']
    clustermodel = args_dict['clustermodel']
    # datafolder = args_dict['datafolder']
    # methodname = args_dict['methodname']
    
    # check that clustertype is either 'hierarchical' or 'kmeans'
    if clustertype not in ['hierarchical', 'kmeans']:
        raise ValueError("clustertype must be either 'hierarchical' or 'kmeans'")
    
    # check that clustermodel is 'linear'
    if clustermodel != 'linear':
        raise ValueError("clustermodel must be 'linear'")
    
    # check that methodname is rf
    # if methodname != 'rf':
    #     raise ValueError("methodname must be 'rf'")
    
    print("Compiling results for " + dataname + " with " + clustertype + \
        " clustering and " + clustermodel + " cluster model")

    # if dataname not in results folder, skip
    if not os.path.exists(f"../lfi-values/seed{seed}/{dataname}"):
        print("No results for " + dataname)
    else:
        
        X = np.loadtxt(oj(dir_data, f"X_{dataname}.csv"), delimiter=",")[1:,:]
        y = np.loadtxt(oj(dir_data, f"y_{dataname}.csv"), delimiter=",")[1:]
        
        # cast to np.float32
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        # split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5,
                                                        random_state = seed)

        glm = ["elastic"]
        ranking = {False: "norank"}

        # create the mapping of variants to argument mappings
        lfi_methods = []
        for g in glm:
            for r in ranking:
                # create the name the variant will be stored under
                variant_name = f"{g}_{ranking[r]}"
                # store the arguments for the lmdi+ explainer
                arg_map = {"glm": g, "ranking": r}
                lfi_methods.append(variant_name)
        lfi_methods.append("lmdi_baseline")

        # for each variant, read in the array
        lfi_value_dict = {}
        for variant in lfi_methods:
            # read in the variant
            lmdi = np.loadtxt(f"../lfi-values/seed{seed}/{dataname}/{variant}.csv", delimiter = ",")
            # get the mse of the variant
            lfi_value_dict[variant] = lmdi
            
        lfi_value_dict["rawdata"] = X_test
        lfi_value_dict["random"] = X_test
        lfi_value_dict["shap"] = np.loadtxt(f"../lfi-values/seed{seed}/{dataname}/shap.csv", delimiter = ",")
        lfi_value_dict["lime"] = np.loadtxt(f"../lfi-values/seed{seed}/{dataname}/lime.csv", delimiter = ",")
        
        # metrics when predicting according to decision tree
        variant_mse_means = []
        variant_mse_sds = []

        for k in range(1, 11):
            
            variant_mse = {}
                
            for variant_name, lmdi_values in lfi_value_dict.items():
                
                # if the variant name is random, create labels randomly by
                # splitting the data into k groups
                if variant_name == "random":
                    np.random.seed(42)
                    labels = np.random.randint(0, k, size=len(y_test))
                
                # perform clustering
                if clustertype == 'kmeans' and variant_name != "random":
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(lmdi_values)
                    labels = kmeans.labels_
                elif clustertype == 'hierarchical' and variant_name != "random":
                    link = hierarchy.linkage(lmdi_values, method='ward')
                    labels = hierarchy.cut_tree(link,n_clusters=k).flatten()

                cluster_mses = np.full((100, k), np.nan)
                cluster_coefs = np.full((100, k, X_test.shape[1]), np.nan)
                cluster_sizes = []
                
                if variant_name == "elastic_norank":
                    # create mappings with the random seeds as keys and a
                    # list of numpy arrays as values
                    global_train_X = defaultdict(list)
                    global_train_y = defaultdict(list)
                    global_test_X = defaultdict(list)
                    global_test_y = defaultdict(list)

                for clust in np.unique(labels):
                    
                    # get the samples in the current cluster
                    cluster_indices = np.where(labels == clust)[0]
                        
                    X_cluster = X_test[cluster_indices]
                    y_cluster = y_test[cluster_indices]
                    cluster_sizes.append(len(cluster_indices))
                                        
                    for rand in range(100):
                        
                        # randomly split the data into train and test (50/50)
                        X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = \
                            train_test_split(X_cluster, y_cluster, test_size=0.5, random_state=rand)
                        
                        # let global model use same train/test split as LMDI+
                        if variant_name == "elastic_norank":
                            # add the train and test data to the lists
                            global_train_X[rand].append(X_train_cluster)
                            global_train_y[rand].append(y_train_cluster)
                            global_test_X[rand].append(X_test_cluster)
                            global_test_y[rand].append(y_test_cluster)
                            
                        # fit cluster model
                        est = LinearRegression()
                        est.fit(X_train_cluster, y_train_cluster)
                        
                        # get coefs
                        cluster_coefs[rand, clust, :] = est.coef_
                                                
                        # get predictions
                        y_pred = est.predict(X_test_cluster)
                        
                        # get performance
                        cluster_mses[rand, clust] = mean_squared_error(y_test_cluster, y_pred)
                    
                if k == 4:
                    result_dir = f"../cluster-results"
                    if not os.path.exists(oj(result_dir, clustertype, clustermodel, dataname, f"seed{seed}")):
                        os.makedirs(oj(result_dir, clustertype, clustermodel, dataname, f"seed{seed}"))
                    # write the cluster labels along with the first two columns of X to csv
                    np.savetxt(f"{result_dir}/{clustertype}/{clustermodel}/{dataname}/seed{seed}/k{k}_{variant_name}_labels.csv", labels, delimiter=",")
                
                if variant_name == "elastic_norank":
                    # combine the train and test data for each seed
                    for key in range(100):
                        global_train_X[key] = np.concatenate(global_train_X[key])
                        global_train_y[key] = np.concatenate(global_train_y[key])
                        global_test_X[key] = np.concatenate(global_test_X[key])
                        global_test_y[key] = np.concatenate(global_test_y[key])
                        
                        # fit model on global data
                        est = LinearRegression()
                        est.fit(global_train_X[key], global_train_y[key])
                        
                        # get predictions
                        y_pred_global = est.predict(global_test_X[key])
                        
                        if key == 0:
                            variant_mse["global_" + variant_name] = [mean_squared_error(global_test_y[key], y_pred_global)]
                        else:
                            variant_mse["global_" + variant_name].append(mean_squared_error(global_test_y[key], y_pred_global))
                    variant_mse["global_" + variant_name] = np.array(variant_mse["global_" + variant_name])
                      
                variant_mse[variant_name] = np.average(cluster_mses, axis=1, weights=cluster_sizes)
                        
            # turn variant_mse into a dataframe with key as column name and mse as value
            variant_mse_df = pd.DataFrame(variant_mse)

            # take the average of each column
            variant_mse_mean = variant_mse_df.mean(axis=0)
            # take the sd of each column
            variant_mse_sd = variant_mse_df.std(axis=0)
            
            # save to list
            variant_mse_means.append(variant_mse_mean)
            variant_mse_sds.append(variant_mse_sd)
            
        # aggregate the list of pd.Series into a dataframe
        variant_mse_means_df = pd.DataFrame(variant_mse_means)
        variant_mse_sds_df = pd.DataFrame(variant_mse_sds)

        # write each of the dataframes to a csv
        # if the path does not exist, create it
        result_dir = f"../cluster-results"
        if not os.path.exists(oj(result_dir, clustertype, clustermodel, dataname, f"seed{seed}")):
            os.makedirs(oj(result_dir, clustertype, clustermodel, dataname, f"seed{seed}"))
        variant_mse_means_df.to_csv(f"{result_dir}/{clustertype}/{clustermodel}/{dataname}/seed{seed}/cluster_mse_mean.csv")
        variant_mse_sds_df.to_csv(f"{result_dir}/{clustertype}/{clustermodel}/{dataname}/seed{seed}/cluster_mse_sd.csv")
        print("Done Writing Cluster Results")
    print("Done Compiling Results")