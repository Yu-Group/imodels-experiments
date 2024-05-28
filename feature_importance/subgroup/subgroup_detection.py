import numpy as np
import pandas as pd
import rbo
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

def compute_rbo_matrix(rankings, p, k=None, side="top", uneven_lengths=False):
    """
    Compute the Rank-based Overlap (RBO) matrix for a set of rankings.
    Inputs:
    * rankings: numpy array of shape (n, p) where the (i,j)th entry denotes that
    the jth feature is ranked (j-1) in terms of importance for the ith instance.
    * p: float between 0 and 1
    * k: int (optional) evaluation depth for extrapolation
    * side: string in {"top", "bottom"} (optional)
    * uneven_lengths: bool (optional)
    Outputs:
    * numpy array of shape (n, n) where the (i,j)th entry is the RBO between the
    ith and jth rankings.
    """
    n, _ = rankings.shape
    rbo_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            rbo_matrix[i, j] = rbo.RankingSimilarity(rankings[i, :], rankings[j, :]).rbo(p=p, k=k)
            rbo_matrix[j, i] = rbo_matrix[i, j]
    return rbo_matrix

def detect_subgroups(rbo_matrix, clustering_method='hierarchical', n_clusters=2,
                     display_plot=True):
    
    # since rbo is a similarity metric, 1 means the rankings are identical
    distance_matrix = rbo_matrix.max()-rbo_matrix

    # convert to condensed distance matrix for scipy compatibility
    condensed_distance_matrix = squareform(distance_matrix)

    if clustering_method == 'hierarchical':
        # perform hierarchical clustering
        Z = linkage(condensed_distance_matrix, method='ward')
        if display_plot:
            # plot the dendrogram
            plt.figure(figsize=(10, 7))
            dendrogram(Z)
            plt.title('Hierarchical Clustering Dendrogram')
            plt.ylabel('Distance')
            ax = plt.gca()
            ax.set_xticks([])
            plt.xlabel('Observations')
            plt.show()
            clusters = fcluster(Z, n_clusters, criterion='maxclust')
            return clusters
    else:
        raise ValueError("Invalid clustering method. Only 'hierarchical' is \
                         supported in this moment.")

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