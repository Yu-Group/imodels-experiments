import numpy as np
from collections import defaultdict

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import BaseEnsemble
import statistics
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import statsmodels.stats.multitest as smt
from tqdm import tqdm
from scripts.util import *


class TreeTester(TransformerMixin, BaseEstimator):

    def __init__(self, estimator, max_components=np.inf, normalize=True):
        self.estimator = estimator
        self.max_components = max_components
        self.normalize = normalize
    
    def get_feature_significance(self,X,y,num_splits = 10):
        p_vals = np.zeros((num_splits,X.shape[1]))
        for i in tqdm(range(num_splits)):
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.5) #perform sample splitting 
            self.estimator.fit(X_train,y_train) #fit on half of sample to learn tree structure and features 
            tree_transformer = TreeTransformer(estimator = self.estimator, max_components=self.max_components) 
            tree_transformer.fit(X_test) 
            transformed_feats = tree_transformer.transform(X_test) #apply tree mapping on X_test 
            OLS_results = sm.OLS(y_test,transformed_feats).fit() #fit decision tree on honest sample
            #OLS_results = sm.OLS(y_test,transformed_feats).fit_regularized(method='elastic_net', alpha=0.1, L1_wt=1.0, start_params=None, profile_scale=False, refit=False)
            for j in range(X.shape[1]):
                num_stumps = transformed_feats.shape[1]
                #P_j = np.zeros((num_stumps,num_stumps))
                num_regressors = len(tree_transformer.original_feat_to_transformed_mapping[j])
                P_j = np.zeros((num_regressors,num_stumps))
                if num_regressors == 0: 
                    p_vals[i,j] = 1.0
                else:
                    for (row_num,feat) in enumerate(tree_transformer.original_feat_to_transformed_mapping[j]):
                    #P_j[feat,feat] = 1
                        P_j[row_num,feat] = 1.0
                    p_vals[i,j] = OLS_results.wald_test(P_j,scalar = False).pvalue
        p_vals[np.isnan(p_vals)] = 1.0
        return np.median(p_vals,axis=0)
    
    def multiple_testing_correction(self,p_vals,method = 'bonferroni',alpha = 0.05):
        return smt.multipletests(p_vals, method=method)[1] 

        
        
            
            
            
    #def get_all_feature_significances(X,y,feat_to_test,num_splits = 10)
            