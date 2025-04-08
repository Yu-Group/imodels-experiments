# standard data science packages
import numpy as np
import pandas as pd

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# hierarchical clustering imports
from scipy.cluster import hierarchy

# data getter imports
from data_loader import load_regr_data

# filesystem imports
import os
from os.path import join as oj

# for command-line args
import argparse

# other
from collections import defaultdict

# datanames = ["Dutch_drinking_inh", "Dutch_drinking_wm", "Dutch_drinking_sha",
#              "Brazil_health_heart", "Brazil_health_stroke", "Korea_grip",
#              "China_glucose_women2", "China_glucose_men2", "Spain_Hair",
#              "China_HIV", "361234", "361235", "361236"]

# openml_ids = ["361234", "361235", "361236", "361237", "361241",
#               "361242", "361243", "361244", "361247", "361249", "361250",
#               "361251", "361252", "361253", "361254", "361255", "361256",
#               "361257", "361258", "361259", "361260", "361261", "361264",
#               "361266", "361267", "361268", "361269", "361272", "361616",
#               "361617", "361618", "361619", "361621", "361622", "361623"]

# dir_data = "../data_openml"

# if __name__ == '__main__':
    
#     # store command-line arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataname', type=str, default=None)
#     parser.add_argument('--seed', type=int, default=None)
#     parser.add_argument('--clustertype', type=str, default=None)
#     parser.add_argument('--clustermodel', type=str, default=None)
#     args = parser.parse_args()
    
#     # convert namespace to a dictionary
#     args_dict = vars(args)

#     # assign the arguments to variables
#     dataname = args_dict['dataname']
#     seed = args_dict['seed']
#     clustertype = args_dict['clustertype']
#     clustermodel = args_dict['clustermodel']
    
#     # check that clustertype is either 'hierarchical' or 'kmeans'
#     if clustertype not in ['hierarchical', 'kmeans']:
#         raise ValueError("clustertype must be either 'hierarchical' or 'kmeans'")
    
#     # check that clustermodel is either 'linear' or 'tree'
#     if clustermodel not in ['linear', 'tree']:
#         raise ValueError("clustermodel must be either 'linear' or 'tree'")
    
#     print("Compiling results for " + dataname + " with " + clustertype + \
#         " clustering and " + clustermodel + " cluster model")

#     # if dataname not in results folder, skip
#     if not os.path.exists(f"../lfi-values/seed{seed}/{dataname}"):
#         print("No results for " + dataname)
#     else:
        
#         X = np.loadtxt(oj(dir_data, f"X_{dataname}.csv"), delimiter=",")[1:,:]
#         y = np.loadtxt(oj(dir_data, f"y_{dataname}.csv"), delimiter=",")[1:]
        
#         # if X has more than 10k rows, sample 10k rows of X and y
#         if X.shape[0] > 10000:
#             np.random.seed(42)
#             indices = np.random.choice(X.shape[0], 10000, replace=False)
#             X = X[indices]
#             y = y[indices]
        
#         # split data into training and testing
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5,
#                                                         random_state = seed)

#         # read in lmdi variants
#         glm = ["ridge", "lasso", "elastic"]
#         normalize = {True: "normed", False: "nonnormed"}
#         square = {True: "squared", False: "nosquared"}
#         leaf_average = {True: "leafavg", False: "noleafavg"}
#         ranking = {True: "rank", False: "norank"}

#         # create the mapping of variants to argument mappings
#         lfi_methods = []
#         for g in glm:
#             for n in normalize:
#                 for s in square:
#                     for r in ranking:
#                         if (not n) and (s):
#                             continue
#                         # create the name the variant will be stored under
#                         variant_name = f"{g}_{normalize[n]}_{square[s]}_{ranking[r]}"
#                         # store the arguments for the lmdi+ explainer
#                         arg_map = {"glm": g, "normalize": n, "square": s,
#                                     "ranking": r}
#                         lfi_methods.append(variant_name)
#         lfi_methods.append("lmdi_baseline")

#         # for each variant, read in the array
#         lfi_value_dict = {}
#         for variant in lfi_methods:
#             # read in the variant
#             lmdi = np.loadtxt(f"../lfi-values/seed{seed}/{dataname}/{variant}.csv", delimiter = ",")
#             # get the mse of the variant
#             lfi_value_dict[variant] = lmdi
            
#         lfi_value_dict["rawdata"] = X_test
#         lfi_value_dict["random"] = X_test
#         lfi_value_dict["shap"] = np.loadtxt(f"../lfi-values/seed{seed}/{dataname}/shap.csv", delimiter = ",")
#         lfi_value_dict["lime"] = np.loadtxt(f"../lfi-values/seed{seed}/{dataname}/lime.csv", delimiter = ",")
        
#         # metrics when predicting according to decision tree
#         variant_mse_means = []
#         variant_mse_sds = []
#         variant_r2_means = []
#         variant_r2_sds = []
        
#         # within cluster variance
#         variant_variance_means = []
#         variant_variance_sds = []
        
#         # metrics when predicting mean of cluster
#         variant_avg_mse_means = []
#         variant_avg_mse_sds = []
#         variant_avg_r2_means = []
#         variant_avg_r2_sds = []

#         for k in range(2, 6):
            
#             variant_mse = {}
#             variant_r2 = {}
#             variant_variance = {}
#             variant_avg_mse = {}
#             variant_avg_r2 = {}
            
#             for rand in range(100):
                
#                 # get a boolean mask that splits the test data into two equal-sized sets
#                 np.random.seed(rand)
#                 is_fit = np.array([True] * (X_test.shape[0] // 2) + [False] * (X_test.shape[0] - (X_test.shape[0] // 2)))
#                 np.random.shuffle(is_fit)
                
#                 if rand < 5:
#                     print(is_fit)
                
#                 for variant_name, lmdi_values in lfi_value_dict.items():
                    
#                     # if the variant name is random, create labels randomly by
#                     # splitting the data into k groups
#                     if variant_name == "random":
#                         np.random.seed(42)
#                         labels = np.random.randint(0, k, size=len(y_test))
                    
#                     # perform clustering
#                     if clustertype == 'kmeans' and variant_name != "random":
#                         kmeans = KMeans(n_clusters=k, random_state=42)
#                         kmeans.fit(lmdi_values)
#                         labels = kmeans.labels_
#                     elif clustertype == 'hierarchical' and variant_name != "random":
#                         link = hierarchy.linkage(lmdi_values, method='ward')
#                         labels = hierarchy.cut_tree(link,n_clusters=k).flatten()

#                     cluster_mse = []
#                     cluster_r2 = []
#                     cluster_sizes = []
#                     cluster_variance = []
#                     cluster_avg_mse = []
#                     cluster_avg_r2 = []
                    
#                     global_train_X = []
#                     global_train_y = []
#                     global_test_X = []
#                     global_test_y = []

#                     for clust in np.unique(labels):
                        
#                         # get the samples in the current cluster
#                         # print(labels.shape)
#                         # print(X_test.shape)
#                         # print(X.shape)
#                         # cluster_indices = np.where(labels == clust)[0]
#                         cluster_mask = labels == clust
                        
#                         # get the samples in the current cluster from the pandas df/series
#                         X_train_cluster = X_test[is_fit & cluster_mask]
#                         y_train_cluster = y_test[is_fit & cluster_mask]
#                         X_test_cluster = X_test[~is_fit & cluster_mask]
#                         y_test_cluster = y_test[~is_fit & cluster_mask]
                        
#                         # X_cluster = X_test[cluster_indices]
#                         # y_cluster = y_test[cluster_indices]
                        
#                         cluster_sizes.append(X_test_cluster.shape[0])
                        
                            
#                         # add the train and test data to the lists
#                         global_train_X.append(X_train_cluster)
#                         global_train_y.append(y_train_cluster)
#                         global_test_X.append(X_test_cluster)
#                         global_test_y.append(y_test_cluster)
                            
#                         # fit cluster model
#                         if clustermodel == 'linear':
#                             est = LinearRegression()
#                         else:
#                             est = DecisionTreeRegressor(max_depth=3,
#                                                         random_state=42)
#                         est.fit(X_train_cluster, y_train_cluster)
                        
#                         # get predictions
#                         y_pred = est.predict(X_test_cluster)
                        
#                         # get performance
#                         cluster_mse.append(mean_squared_error(y_test_cluster, y_pred))
#                         cluster_r2.append(r2_score(y_test_cluster, y_pred))
#                         cluster_variance.append(np.var(y_test_cluster))
                        
#                     # combine the train and test data
#                     global_train_X = np.concatenate(global_train_X)
#                     global_train_y = np.concatenate(global_train_y)
#                     global_test_X = np.concatenate(global_test_X)
#                     global_test_y = np.concatenate(global_test_y)
                                        
#                     # fit model on global data
#                     if clustermodel == 'linear':
#                         est = LinearRegression()
#                     else:
#                         est = DecisionTreeRegressor(max_depth=3,
#                                                     random_state=42)
#                     est.fit(global_train_X, global_train_y)
                    
#                     # get predictions
#                     y_pred_global = est.predict(global_test_X)
                    
                    
#                     if variant_name not in variant_mse:
#                         variant_mse[variant_name] = [np.average(cluster_mse, weights=cluster_sizes)]
#                         variant_mse["global_" + variant_name] = [mean_squared_error(global_test_y, y_pred_global)]
#                     else:
#                         variant_mse[variant_name].append(np.average(cluster_mse, weights=cluster_sizes))
#                         variant_mse["global_" + variant_name].append(mean_squared_error(global_test_y, y_pred_global))
#                     if variant_name not in variant_r2:
#                         variant_r2[variant_name] = [np.average(cluster_r2, weights=cluster_sizes)]
#                         variant_r2["global_" + variant_name] = [r2_score(global_test_y, y_pred_global)]
#                     else:
#                         variant_r2[variant_name].append(np.average(cluster_r2, weights=cluster_sizes))
#                         variant_r2["global_" + variant_name].append(r2_score(global_test_y, y_pred_global))
#                     if variant_name not in variant_variance:
#                         variant_variance[variant_name] = [np.average(cluster_variance, weights=cluster_sizes)]
#                         variant_variance["global_" + variant_name] = [np.var(global_test_y)]
#                     else:
#                         variant_variance[variant_name].append(np.average(cluster_variance, weights=cluster_sizes))
#                         variant_variance["global_" + variant_name].append(np.var(global_test_y))
                        
#             # turn variant_mse into a dataframe with key as column name and mse as value
#             variant_mse_df = pd.DataFrame(variant_mse)
#             # take the average of each column
#             variant_mse_mean = variant_mse_df.mean(axis=0)
#             # take the sd of each column
#             variant_mse_sd = variant_mse_df.std(axis=0)
            
#             # save to list
#             variant_mse_means.append(variant_mse_mean)
#             variant_mse_sds.append(variant_mse_sd)
            
#             # turn variant_r2 into a dataframe with key as column name and mse as value
#             variant_r2_df = pd.DataFrame(variant_r2)
#             # take the average of each column
#             variant_r2_mean = variant_r2_df.mean(axis=0)
#             # take the sd of each column
#             variant_r2_sd = variant_r2_df.std(axis=0)
            
#             # save to list
#             variant_r2_means.append(variant_r2_mean)
#             variant_r2_sds.append(variant_r2_sd)
            
#             # turn variant_variance_means into a dataframe with key as column name and mse as value
#             variant_variance_means_df = pd.DataFrame(variant_variance)
#             # take the average of each column
#             variant_variance_mean = variant_variance_means_df.mean(axis=0)
#             # take the sd of each column
#             variant_variance_sd = variant_variance_means_df.std(axis=0)

#             # save to list
#             variant_variance_means.append(variant_variance_mean)
#             variant_variance_sds.append(variant_variance_sd)
            
#         # aggregate the list of pd.Series into a dataframe
#         variant_mse_means_df = pd.DataFrame(variant_mse_means)
#         variant_mse_sds_df = pd.DataFrame(variant_mse_sds)
#         variant_r2_means_df = pd.DataFrame(variant_r2_means)
#         variant_r2_sds_df = pd.DataFrame(variant_r2_sds)
#         variant_variance_means_df = pd.DataFrame(variant_variance_means)
#         variant_variance_sds_df = pd.DataFrame(variant_variance_sds)

#         # write each of the dataframes to a csv
#         # if the path does not exist, create it
#         result_dir = "../cluster-results/split-pre-cluster"
#         if not os.path.exists(oj(result_dir, clustertype, clustermodel, dataname, f"seed{seed}")):
#             os.makedirs(oj(result_dir, clustertype, clustermodel, dataname, f"seed{seed}"))
#         variant_mse_means_df.to_csv(f"{result_dir}/{clustertype}/{clustermodel}/{dataname}/seed{seed}/cluster_mse_mean.csv")
#         variant_mse_sds_df.to_csv(f"{result_dir}/{clustertype}/{clustermodel}/{dataname}/seed{seed}/cluster_mse_sd.csv")
#         variant_r2_means_df.to_csv(f"{result_dir}/{clustertype}/{clustermodel}/{dataname}/seed{seed}/cluster_r2_mean.csv")
#         variant_r2_sds_df.to_csv(f"{result_dir}/{clustertype}/{clustermodel}/{dataname}/seed{seed}/cluster_r2_sd.csv")
#         variant_variance_means_df.to_csv(f"{result_dir}/{clustertype}/{clustermodel}/{dataname}/seed{seed}/cluster_variance_mean.csv")
#         variant_variance_sds_df.to_csv(f"{result_dir}/{clustertype}/{clustermodel}/{dataname}/seed{seed}/cluster_variance_sd.csv")

#         print("Done Writing Cluster Results")
            

#     print("Done Compiling Results")

dir_data = "../data_openml"

if __name__ == '__main__':
    
    # store command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--clustertype', type=str, default=None)
    parser.add_argument('--clustermodel', type=str, default=None)
    parser.add_argument('--datafolder', type=str, default=None)
    parser.add_argument('--methodname', type=str, default=None)
    args = parser.parse_args()
    
    # convert namespace to a dictionary
    args_dict = vars(args)

    # assign the arguments to variables
    dataname = args_dict['dataname']
    seed = args_dict['seed']
    clustertype = args_dict['clustertype']
    clustermodel = args_dict['clustermodel']
    datafolder = args_dict['datafolder']
    methodname = args_dict['methodname']
    
    # check that clustertype is either 'hierarchical' or 'kmeans'
    if clustertype not in ['hierarchical', 'kmeans']:
        raise ValueError("clustertype must be either 'hierarchical' or 'kmeans'")
    
    # check that clustermodel is either 'linear' or 'tree'
    if clustermodel not in ['linear', 'tree']:
        raise ValueError("clustermodel must be either 'linear' or 'tree'")
    
    # check that methodname is either rf or gb
    if methodname not in ['rf', 'gb']:
        raise ValueError("methodname must be either 'rf' or 'gb'")
    
    print("Compiling results for " + dataname + " with " + clustertype + \
        " clustering and " + clustermodel + " cluster model")

    # if dataname not in results folder, skip
    if not os.path.exists(f"../lfi-values/{datafolder}/{methodname}/seed{seed}/{dataname}"):
        print("No results for " + dataname)
    else:
        
        X = np.loadtxt(oj(dir_data, f"X_{dataname}.csv"), delimiter=",")[1:,:]
        y = np.loadtxt(oj(dir_data, f"y_{dataname}.csv"), delimiter=",")[1:]
        
        # cast to np.float32
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        # if the data is standardize it, we need to standardize again here
        if datafolder == "standardized-fulldata":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            y = (y - np.mean(y)) / np.std(y)
        if datafolder == "standardizedX-fulldata":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        # if X has more than 5k rows, sample 5k rows of X and y
        # if X.shape[0] > 5000:
        #     np.random.seed(42)
        #     indices = np.random.choice(X.shape[0], 5000, replace=False)
        #     X = X[indices]
        #     y = y[indices]
        
        # split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5,
                                                        random_state = seed)

        
        # X, y, names_covariates = load_regr_data(dataname, dir_data)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
        #                                                     random_state = seed)
        # read in lmdi variants
        glm = ["ridge", "lasso", "elastic"]
        normalize = {True: "normed", False: "nonnormed"}
        square = {True: "squared", False: "nosquared"}
        leaf_average = {True: "leafavg", False: "noleafavg"}
        ranking = {True: "rank", False: "norank"}

        # create the mapping of variants to argument mappings
        lfi_methods = []
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
                        lfi_methods.append(variant_name)
        lfi_methods.append("lmdi_baseline")

        # for each variant, read in the array
        lfi_value_dict = {}
        for variant in lfi_methods:
            # read in the variant
            lmdi = np.loadtxt(f"../lfi-values/{datafolder}/{methodname}/seed{seed}/{dataname}/{variant}.csv", delimiter = ",")
            # get the mse of the variant
            lfi_value_dict[variant] = lmdi
            
        lfi_value_dict["rawdata"] = X_test
        lfi_value_dict["random"] = X_test
        lfi_value_dict["shap"] = np.loadtxt(f"../lfi-values/{datafolder}/{methodname}/seed{seed}/{dataname}/shap.csv", delimiter = ",")
        lfi_value_dict["lime"] = np.loadtxt(f"../lfi-values/{datafolder}/{methodname}/seed{seed}/{dataname}/lime.csv", delimiter = ",")
        
        # metrics when predicting according to decision tree
        variant_mse_means = []
        variant_mse_sds = []
        # variant_r2_means = []
        # variant_r2_sds = []
        
        # within cluster variance
        # variant_variance_means = []
        # variant_variance_sds = []
        
        # metrics when predicting mean of cluster
        # variant_avg_mse_means = []
        # variant_avg_mse_sds = []
        # variant_avg_r2_means = []
        # variant_avg_r2_sds = []
        
        # k_size_info_maps = {}

        for k in range(2, 11):
            
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
                
                if variant_name == "elastic_nonnormed_nosquared_norank":
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
                            
                        if variant_name == "elastic_nonnormed_nosquared_norank":
                            # add the train and test data to the lists
                            global_train_X[rand].append(X_train_cluster)
                            global_train_y[rand].append(y_train_cluster)
                            global_test_X[rand].append(X_test_cluster)
                            global_test_y[rand].append(y_test_cluster)
                            
                        # fit cluster model
                        if clustermodel == 'linear':
                            est = LinearRegression()
                        else:
                            est = DecisionTreeRegressor(max_depth=3,
                                                        random_state=42)
                        est.fit(X_train_cluster, y_train_cluster)
                        
                        # get coefs
                        cluster_coefs[rand, clust, :] = est.coef_
                                                
                        # get predictions
                        y_pred = est.predict(X_test_cluster)
                        
                        # get performance
                        cluster_mses[rand, clust] = mean_squared_error(y_test_cluster, y_pred)
                    
                    # average the cluster coefs
                    cluster_coefs_avg = np.mean(cluster_coefs, axis=0)
                    # if k == 5:
                    #     if seed == 0:
                    #         result_dir = f"../cluster-results/{methodname}"
                    #         if not os.path.exists(oj(result_dir, clustertype, clustermodel, dataname, f"seed{seed}")):
                    #             os.makedirs(oj(result_dir, clustertype, clustermodel, dataname, f"seed{seed}"))
                    #         # write the cluster labels along with the first two columns of X to csv
                    #         np.savetxt(f"{result_dir}/{clustertype}/{clustermodel}/{dataname}/seed{seed}/{k}clusters_clust{clust}_{variant_name}_coefs.csv", cluster_coefs_avg, delimiter=",")
                if seed==0 and k == 4:
                    result_dir = f"../cluster-results/{methodname}"
                    if not os.path.exists(oj(result_dir, clustertype, clustermodel, dataname, f"seed{seed}")):
                        os.makedirs(oj(result_dir, clustertype, clustermodel, dataname, f"seed{seed}"))
                    # write the cluster labels along with the first two columns of X to csv
                    np.savetxt(f"{result_dir}/{clustertype}/{clustermodel}/{dataname}/seed{seed}/k{k}_{variant_name}_labels.csv", labels, delimiter=",")
                
                if variant_name == "elastic_nonnormed_nosquared_norank":
                    # combine the train and test data for each seed
                    for key in range(100):
                        global_train_X[key] = np.concatenate(global_train_X[key])
                        global_train_y[key] = np.concatenate(global_train_y[key])
                        global_test_X[key] = np.concatenate(global_test_X[key])
                        global_test_y[key] = np.concatenate(global_test_y[key])
                        
                        # fit model on global data
                        if clustermodel == 'linear':
                            est = LinearRegression()
                        else:
                            est = DecisionTreeRegressor(max_depth=3,
                                                        random_state=42)
                        est.fit(global_train_X[key], global_train_y[key])
                        
                        # get predictions
                        y_pred_global = est.predict(global_test_X[key])
                        
                        if key == 0:
                            variant_mse["global_" + variant_name] = [mean_squared_error(global_test_y[key], y_pred_global)]
                        else:
                            variant_mse["global_" + variant_name].append(mean_squared_error(global_test_y[key], y_pred_global))
                    variant_mse["global_" + variant_name] = np.array(variant_mse["global_" + variant_name])
                    print(variant_mse["global_" + variant_name])
                      
                variant_mse[variant_name] = np.average(cluster_mses, axis=1, weights=cluster_sizes)
            # print(variant_mse)
                        
            # turn variant_mse into a dataframe with key as column name and mse as value
            variant_mse_df = pd.DataFrame(variant_mse)
            # print(variant_mse_df.shape)
            # take the average of each column
            variant_mse_mean = variant_mse_df.mean(axis=0)
            # take the sd of each column
            # print(variant_mse_df.shape)
            # print(variant_mse_df)
            variant_mse_sd = variant_mse_df.std(axis=0)
            
            # save to list
            variant_mse_means.append(variant_mse_mean)
            variant_mse_sds.append(variant_mse_sd)
            
        # aggregate the list of pd.Series into a dataframe
        variant_mse_means_df = pd.DataFrame(variant_mse_means)
        variant_mse_sds_df = pd.DataFrame(variant_mse_sds)
        # print(variant_mse_means_df)
        # print(variant_mse_sds_df)

        # write each of the dataframes to a csv
        # if the path does not exist, create it
        # result_dir = f"../cluster-results/{datafolder}/{methodname}/split-post-cluster"
        result_dir = f"../cluster-results/{methodname}"
        if not os.path.exists(oj(result_dir, clustertype, clustermodel, dataname, f"seed{seed}")):
            os.makedirs(oj(result_dir, clustertype, clustermodel, dataname, f"seed{seed}"))
        variant_mse_means_df.to_csv(f"{result_dir}/{clustertype}/{clustermodel}/{dataname}/seed{seed}/cluster_mse_mean.csv")
        variant_mse_sds_df.to_csv(f"{result_dir}/{clustertype}/{clustermodel}/{dataname}/seed{seed}/cluster_mse_sd.csv")
        print("Done Writing Cluster Results")
    print("Done Compiling Results")