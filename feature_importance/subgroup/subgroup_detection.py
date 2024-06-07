import numpy as np
import pandas as pd
import rbo
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

def compute_rbo_matrix(rankings, p, k=None):
    """
    Compute the Rank-based Overlap (RBO) matrix for a set of rankings.
    Inputs:
    * rankings: numpy array of shape (n, p) where the (i,j)th entry denotes that
    the jth feature is ranked (j-1) in terms of importance for the ith instance.
    * p: float between 0 and 1
    * k: int (optional) evaluation depth for extrapolation
    Outputs:
    * numpy array of shape (n, n) where the (i,j)th entry is the RBO between the
    ith and jth rankings.
    """
    n, _ = rankings.shape
    rbo_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            rbo_matrix[i, j] = rbo.RankingSimilarity(rankings[i, :],
                                                     rankings[j,:]).rbo(p=p,k=k)
            rbo_matrix[j, i] = rbo_matrix[i, j]
    return rbo_matrix

def detect_subgroups(mdi, rankings, num_clusters, p = 0.9, k = None,
                     linkage_method = 'ward'):
        
    # compute rbo matrix
    rbo_matrix = compute_rbo_matrix(rankings, 0.9, k=k)
        
    # since rbo is a similarity metric, 1 means the rankings are identical
    distance_matrix = rbo_matrix.max()-rbo_matrix

    # convert to condensed distance matrix for scipy compatibility
    condensed_distance_matrix = squareform(distance_matrix)

    # perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance_matrix, method="ward")
    clustergrid = sns.clustermap(mdi, row_linkage=linkage_matrix,
                                 col_cluster=False, cmap='viridis',
                                 cbar_pos = (1, 0.2, 0.05, 0.5))
    
    # Get the reordered row indices
    reordered_indices = clustergrid.dendrogram_row.reordered_ind

    # Determine cluster memberships using fcluster
    clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

    # Reorder clusters to match the heatmap
    ordered_clusters = clusters[reordered_indices]
    
    # Create a new DataFrame for annotations
    annotations = pd.DataFrame(data=np.zeros_like(mdi, dtype=object), index=mdi.index, columns=mdi.columns)

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
        x_position = mdi.shape[1] - 3  # Adjust this value for correct alignment
        y_position = total_obs + num_obs//2  # Adjust this value for correct alignment
        ax.text(x_position, y_position, "Cluster #" + str(cluster), color='red', ha='center', va='center', fontsize=10, fontweight='bold')
        total_obs += num_obs

    plt.suptitle('Heatmap with Hierarchical Clustering', fontsize=24)
    plt.show()
    
    return clusters

# def one_hot_encode(rankings):
#     """
#     Obtain a one-hot encoded representation of the local feature importance
#     rankings.
#     Inputs:
#     * rankings: numpy array of shape (n, p) where the (i,j)th entry denotes that
#     the jth feature is ranked (j-1) in terms of importance for the ith instance.
#     Outputs:
#     * numpy array of shape (n, p^2) which encodes the same information as the
#     input, but formatted as a matrix with binary entries.
#     """
#     n, p = rankings.shape
#     one_hot_matrix = np.full(shape = (n, p**2), fill_value = -1, dtype = int)
#     for i in range(n):
#         one_hot_matrix[i, :] = np.array(pd.get_dummies(rankings[0,:]),
#                                         dtype = int).flatten()
#     return one_hot_matrix

# def decision_tree_grouper(rankings, y):
#     """
#     Group features based on the decision tree feature importance rankings.
#     Inputs:
#     * rankings: numpy array of shape (n, p) where the (i,j)th entry denotes that
#     the jth feature is ranked (j-1) in terms of importance for the ith instance.
#     * y: length n array of the target labels.
#     Outputs:
#     * length n array where the ith entry is the group # of the ith instance.
#     """
#     encoded_rankings = one_hot_encode(rankings)
#     # check if y is binary or continuous
#     if len(np.unique(y)) == 2:
#         clf = DecisionTreeClassifier()
#         print("classification tree fit")
#     else:
#         clf = DecisionTreeRegressor()
#         print("regression tree fit")
#     clf.fit(encoded_rankings, y)
    
#     # get which leaf each instance falls into
#     return clf.apply(encoded_rankings)