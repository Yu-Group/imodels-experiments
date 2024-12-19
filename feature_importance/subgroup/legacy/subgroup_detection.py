import numpy as np
import pandas as pd
import rbo
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

def weighted_metric(metrics, sample_sizes):
    """
    Calculate the weighted average of a set of metrics.
    
    Args:
        sample_sizes (np.ndarray): the number of samples in each subgroup
        metrics (np.ndarray): the metrics of each subgroup
        
    Returns:
        float: the weighted average of the metrics
    """
    
    print("Sample sizes: ", sample_sizes)
    print("Metrics: ", metrics)
    print("Sample sizes type: ", type(sample_sizes))
    print("Metrics type: ", type(metrics))
    
    # calculate the total number of samples
    total_samples = np.sum(sample_sizes)
    
    # calculate the weighted average
    weighted_metric = np.sum(sample_sizes * metrics) / total_samples
    
    return weighted_metric

def compute_rbo_matrix(rankings, form, p=0.9, k=None, ext=False):
    """
    Compute the distance matrix based on Rank-based Overlap (RBO).
    Inputs:
    * rankings: numpy array of shape (n, p) where the (i,j)th entry denotes that
    the jth feature is ranked (j-1) in terms of importance for the ith instance.
    * p: float between 0 and 1
    * k: int (optional) evaluation depth for extrapolation
    * ext: bool (optional) whether to use extrapolation
    Outputs:
    * numpy array of shape (n, n) where the (i,j)th entry is the RBO between the
    ith and jth rankings.
    """
    # ensure form is either "distance" or "similarity"
    if form not in ['distance', 'similarity']:
        raise ValueError('form must be either "distance" or "similarity"')
    n = rankings.shape[0]
    rbo_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            rbo_matrix[i, j] = rbo.RankingSimilarity(rankings[i,:],
                                                     rankings[j,:]).rbo(p=p,k=k,
                                                                        ext=ext)
            rbo_matrix[j, i] = rbo_matrix[i, j]
    
    if form == "distance":
        # since rbo is a similarity metric, 1 means the rankings are identical
        return rbo_matrix.max()-rbo_matrix
    else:
        return rbo_matrix

def assign_training_clusters(rbo_distance_matrix, num_clusters,
                             linkage_method='ward'):

    # convert to condensed distance matrix for scipy compatibility
    condensed_distance_matrix = squareform(rbo_distance_matrix)

    # perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance_matrix, method=linkage_method)

    # Determine cluster memberships using fcluster
    clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    
    return clusters

def plot_training_clusters(lfi, rbo_distance_matrix,
                           clusters, linkage_method='ward'):
    
    # convert to condensed distance matrix for scipy compatibility
    condensed_distance_matrix = squareform(rbo_distance_matrix)

    # perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance_matrix, method=linkage_method)
    clustergrid = sns.clustermap(lfi, row_linkage=linkage_matrix,
                                 col_cluster=False, cmap='viridis',
                                 cbar_pos = (1, 0.2, 0.05, 0.5))
    
    # Get the reordered row indices
    reordered_indices = clustergrid.dendrogram_row.reordered_ind

    # determine number of clusters
    num_clusters = np.unique(clusters).shape[0]
        
    # Reorder clusters to match the heatmap
    ordered_clusters = clusters[reordered_indices]
    
    # Create a new DataFrame for annotations
    annotations = pd.DataFrame(data=np.zeros_like(lfi, dtype=object),
                            index=lfi.index, columns=lfi.columns)

    # Add cluster numbers as annotations
    for i, cluster in enumerate(ordered_clusters):
        annotations.iloc[i, :] = cluster

    # Find the boundaries where clusters change
    boundaries = np.where(np.diff(ordered_clusters))[0] + 1

    # Plot the horizontal dashed red lines
    for boundary in boundaries:
        clustergrid.ax_heatmap.hlines(boundary,
                                    *clustergrid.ax_heatmap.get_xlim(),
                                    colors='red', linestyles='dashed')
        
    ax = clustergrid.ax_heatmap
    total_obs = 0
    for cluster in range(1, num_clusters + 1):
        num_obs = np.sum(clusters == cluster)
        x_position = lfi.shape[1]//2  # for alignment
        y_position = total_obs + num_obs//2  # for alignment
        ax.text(x_position, y_position, "Cluster #" + str(cluster), color='red',
                ha='center', va='center', fontsize=10, fontweight='bold')
        total_obs += num_obs

    
    plt.suptitle('Heatmap with Hierarchical Clustering', fontsize=24)
    return

def find_geometric_median(distance_matrix):
    
    # calculate the sum of distances for each point
    distance_sums = np.sum(distance_matrix, axis=1)
    
    # find the index of the point with the minimum sum of distances
    geometric_median_index = np.argmin(distance_sums)
    
    return geometric_median_index

def assign_testing_centroid_approx(rbo_distance_matrix: np.ndarray,
                                   lfi_train_ranking: np.ndarray,
                                   lfi_test_ranking: np.ndarray,
                                   clusters: np.ndarray) -> np.ndarray:
    """
    Assigns testing points to clusters based on similarity to median ranking.

    Args:
        rbo_distance_matrix (np.ndarray): distance matrix based on the rbo
        lfi_train_ranking (np.ndarray): the local feature importance rankings of
            the training points
        lfi_test_ranking (np.ndarray): the local feature importance rankings of
            the testing points
        clusters (np.ndarray): the cluster labels for the training points

    Returns:
        np.ndarray: cluster lables for the testing points
    """

    # initialize array to store cluster assignments for testing points
    test_clust = np.zeros(lfi_test_ranking.shape[0])
    
    # iterate over testing points and
    # assign to cluster with most similar median ranking
    for i in range(lfi_test_ranking.shape[0]):
        king_of_hill = float('-inf') # start at negative infinity
        # loop through clusters, calculate median ranking, and compare to point
        for j in range(np.unique(clusters).shape[0]):
            # get distance matrix for just this cluster
            rbo_distance_clust = \
                rbo_distance_matrix[clusters == j+1, :][:, clusters == j+1]
            median_index = find_geometric_median(rbo_distance_clust)
            median_ranking = lfi_train_ranking[clusters==j+1,:][median_index,:]
            similarity = rbo.RankingSimilarity(median_ranking,
                                    lfi_test_ranking[i, :]).rbo(p=0.9, k=None)
            if similarity > king_of_hill:
                king_of_hill = similarity
                test_clust[i] = j+1
    return test_clust

def assign_testing_centroid_exact(rbo_distance_matrix: np.ndarray,
                                  lfi_train_ranking: np.ndarray,
                                  lfi_test_ranking: np.ndarray,
                                  clusters: np.ndarray) -> np.ndarray:
    """
    Assigns testing points to clusters based on similarity to average ranking.

    Args:
        rbo_distance_matrix (np.ndarray): distance matrix based on the rbo
        lfi_train_ranking (np.ndarray): the local feature importance rankings of
            the training points
        lfi_test_ranking (np.ndarray): the local feature importance rankings of
            the testing points
        clusters (np.ndarray): the cluster labels for the training points

    Returns:
        np.ndarray: cluster lables for the testing points
    """
    
    # initialize array to store cluster assignments for testing points
    test_clust = np.zeros(lfi_test_ranking.shape[0])
    
    # iterate over testing points and calculate rbo similarity to each cluster
    # assign to cluster with most similar points (on average)
    for i in range(lfi_test_ranking.shape[0]):
        king_of_hill = float('-inf') # start at negative infinity
        # loop through clusters
        for j in range(np.unique(clusters).shape[0]):
            # get distance matrix for just this cluster
            curr_clust = lfi_train_ranking[clusters == j+1, :]
            similarity = np.zeros(curr_clust.shape[0])
            # compute similarity to each point in cluster
            for k in range(curr_clust.shape[0]):
                similarity[k] = rbo.RankingSimilarity(curr_clust[k, :],
                                    lfi_test_ranking[i, :]).rbo(p=0.9, k=None)
            # assign to cluster with most similar rankings on average
            if similarity.mean() > king_of_hill:
                king_of_hill = similarity.mean()
                test_clust[i] = j+1
                
    return test_clust

def within_cluster_variance(rbo_distance_matrix: np.ndarray):
    """
    Calculates the variance of a set of points given a distance matrix.

    Args:
        rbo_distance_matrix (numpy.ndarray): a square matrix representing the
        pairwise distances between points.

    Returns:
        float: The variance of the set of points.
    """
    n = rbo_distance_matrix.shape[0]
    
    # calculate the mean distance
    mean_distance = np.mean(rbo_distance_matrix)
    
    # calculate the squared differences
    squared_diffs = (rbo_distance_matrix - mean_distance)**2
    
    # sum squared differences, divide by the total number of pairwise distances
    variance = np.sum(squared_diffs) / (n * (n - 1))
    
    return variance

def rbo_distance_offset(num_features):
    return rbo.RankingSimilarity(np.arange(1, num_features+1),
                                 np.arange(1, num_features+1)).rbo(p=0.9,k=None)

def assign_testing_variance_exact(rbo_distance_matrix: np.ndarray,
                                   lfi_train_ranking: np.ndarray,
                                   lfi_test_ranking: np.ndarray,
                                   clusters: np.ndarray) -> np.ndarray:
    """
    Assigns testing points to clusters based on smallest variance increase.

    Args:
        rbo_distance_matrix (np.ndarray): distance matrix based on the rbo
        lfi_train_ranking (np.ndarray): the local feature importance rankings of
            the training points
        lfi_test_ranking (np.ndarray): the local feature importance rankings of
            the testing points
        clusters (np.ndarray): the cluster labels for the training points

    Returns:
        np.ndarray: cluster lables for the testing points
    """
    
    # initialize array to store cluster assignments for testing points
    test_clust = np.zeros(lfi_test_ranking.shape[0])
    
    # iterate over testing points and
    # assign to cluster with smallest variance increase
    for i in range(lfi_test_ranking.shape[0]):
        king_of_hill = float('inf') # start at negative infinity
        # loop through clusters, calculate median ranking, and compare to point
        for j in range(np.unique(clusters).shape[0]):
            # get distance matrix for just this cluster
            rbo_distance_clust = \
                rbo_distance_matrix[clusters == j+1, :][:, clusters == j+1]
            current_variance = within_cluster_variance(rbo_distance_clust)
            distance_to_point = np.zeros(rbo_distance_clust.shape[0])
            for k in range(rbo_distance_clust.shape[0]):
                distance_to_point[k] = \
                    rbo_distance_offset(lfi_test_ranking.shape[1]) - \
                        rbo.RankingSimilarity(
                            lfi_train_ranking[clusters == j+1, :][k, :],
                            lfi_test_ranking[i,:]).rbo(p=0.9, k=None)
            # add distance_to_point as new row and column, diagonal elem is zero
            extended_rbo_distance_clust = np.zeros((rbo_distance_clust.shape[0]+1,
                                                    rbo_distance_clust.shape[1]+1))
            extended_rbo_distance_clust[:-1, :-1] = rbo_distance_clust
            # add distance_to_point as last row and column
            extended_rbo_distance_clust[-1, :-1] = distance_to_point
            extended_rbo_distance_clust[:-1, -1] = distance_to_point
            
            new_variance = within_cluster_variance(extended_rbo_distance_clust)
            variance_increase = new_variance - current_variance
            if variance_increase < king_of_hill:
                king_of_hill = variance_increase
                test_clust[i] = j+1
    return test_clust

def assign_testing_clusters(method: str, median_approx: bool,
                            rbo_distance_matrix: np.ndarray,
                            lfi_train_ranking: np.ndarray,
                            lfi_test_ranking: np.ndarray,
                            clusters: np.ndarray) -> np.ndarray:
    """
    Assigns testing points to clusters based on the method specified.

    Args:
        method (str): the method to use for assigning testing points to clusters
        median_approx (bool): whether to use the median approximation
        rbo_distance_matrix (np.ndarray): distance matrix based on the rbo
        lfi_train_ranking (np.ndarray): the local feature importance rankings of
            the training points
        lfi_test_ranking (np.ndarray): the local feature importance rankings of
            the testing points
        clusters (np.ndarray): the cluster labels for the training points

    Raises:
        ValueError: if the method is not centroid or variance

    Returns:
        np.ndarray: cluster lables for the testing points
    """
    
    # ensure the method is either centroid or variance
    if method not in ['centroid', 'variance']:
        raise ValueError('method must be either centroid or variance')
    
    # compute the testing point cluster assignments as specified by arguments
    if method == "centroid":
        if median_approx:
            return assign_testing_centroid_approx(rbo_distance_matrix,
                                                  lfi_train_ranking,
                                                  lfi_test_ranking, clusters)
        else:
            return assign_testing_centroid_exact(rbo_distance_matrix,
                                                 lfi_train_ranking,
                                                 lfi_test_ranking, clusters)
    else:
        if median_approx:
            return assign_testing_variance_exact(rbo_distance_matrix,
                                                 lfi_train_ranking,
                                                 lfi_test_ranking, clusters)
        else:
            return assign_testing_variance_exact(rbo_distance_matrix,
                                                 lfi_train_ranking,
                                                 lfi_test_ranking, clusters)
            
def match_subgroups(cluster1, cluster2):
    """
    Match subgroups between two clusterings.
    
    Args:
        cluster1 (np.ndarray): cluster assignments for the first clustering
        cluster2 (np.ndarray): cluster assignments for the second clustering
        
    Returns:
        np.ndarray: an array of shape (num_clusters1,) where the ith entry is
        the cluster number in the second clustering that best matches the ith
        cluster in the first clustering
    """
    
    # create storage dictionaries
    dict1to2 = {}
    dict2to1 = {}
    
    # find the number of clusters in each clustering
    num_clusters1 = np.unique(cluster1).shape[0]
    num_clusters2 = np.unique(cluster2).shape[0]
    
    # check that each cluster has at least one point
    if num_clusters1 != num_clusters2:
        print("Warning: Number of clusters in clusterings do not match. Returning None.")
        return None
    
    for clust1 in range(1, num_clusters1 + 1):
        # get the points in cluster 1
        points1 = np.where(cluster1 == clust1)[0]
        # for each cluster in the second clustering, find the percentage of its points in points1
        best_percentage = 0
        best_match = None
        for clust2 in range(1, num_clusters2 + 1):
            # get the points in cluster 2
            points2 = np.where(cluster2 == clust2)[0]
            # calculate the percentage of points in points2 that are in points1
            percentage = len(np.intersect1d(points1, points2)) / len(points2)
            if percentage > best_percentage:
                best_percentage = percentage
                best_match = clust2
        dict1to2[clust1] = best_match
    for clust2 in range(1, num_clusters2 + 1):
        # get the points in cluster 2
        points2 = np.where(cluster2 == clust2)[0]
        # for each cluster in the first clustering, find the percentage of its points in points2
        best_percentage = 0
        best_match = None
        for clust1 in range(1, num_clusters1 + 1):
            # get the points in cluster 1
            points1 = np.where(cluster1 == clust1)[0]
            # calculate the percentage of points in points1 that are in points2
            percentage = len(np.intersect1d(points1, points2)) / len(points1)
            if percentage > best_percentage:
                best_percentage = percentage
                best_match = clust1
        dict2to1[clust2] = best_match
    print("dict1to2")
    print(dict1to2)
    print("dict2to1")
    print(dict2to1)
    # check that the dictionaries agree with each other
    for clust1 in range(1, num_clusters1 + 1):
        if dict2to1[dict1to2[clust1]] != clust1:
            print("Warning: Dictionaries do not agree with each other. Returning None.")
            return None
    
    converted_membership = [dict2to1[cls] for cls in cluster2]
    return converted_membership
    
    