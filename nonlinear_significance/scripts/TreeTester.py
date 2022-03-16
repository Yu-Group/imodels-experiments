import numpy as np
from collections import defaultdict

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import BaseEnsemble
import statistics
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
import statsmodels.stats.multitest as smt
from tqdm import tqdm
import torch 
import sys,os
from torch import nn
from numpy import linalg as LA
from torch.functional import F
from copy import copy
from sklearn.model_selection import GridSearchCV
from torch.autograd import Variable

#from nonlinear_significance.scripts.util import *
from nonlinear_significance.scripts.util import TreeTransformer
from util import *

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
        median_p_vals = 2*np.median(p_vals,axis=0)
        median_p_vals[median_p_vals > 1.0] = 1.0
        #return median_p_vals
        return self.multiple_testing_correction(median_p_vals)


    
    def multiple_testing_correction(self,p_vals,method = 'bonferroni',alpha = 0.05):
        return smt.multipletests(p_vals, method=method)[1] 

    
class optimalTreeTester(TransformerMixin, BaseEstimator): #This class is trying to improve the power of TreeTester by implementing an optimal weighting scheme that favors big nodes...

    def __init__(self, estimator, normalize=True):
        self.estimator = estimator
        self.normalize = normalize
    
    def get_feature_significance(self,X,y,num_splits = 10,eta = None,lr = .1,n_steps = 3000,num_reps = 10000,max_components = 0.5,params = {}):
        p_vals = np.ones((num_splits,X.shape[1]))
        for i in tqdm(range(num_splits)):
            X_sel, X_inf, y_sel, y_inf = train_test_split(X,y,test_size = 0.5)
            
            gs_estimator = GridSearchCV(self.estimator,param_grid = params, scoring = 'r2', cv = 5)
            gs_estimator.fit(X_sel, y_sel)
            
            self.estimator = gs_estimator.best_estimator_
            
            
            self.estimator.fit(X_sel,y_sel) #fit on half of sample to learn tree structure and features 
                        
            
            tree_transformer_sel = TreeTransformer(estimator = self.estimator, max_components= int(X_sel.shape[0]*max_components) )
            tree_transformer_sel.fit(X_sel) 
            transformed_feats_sel = tree_transformer_sel.transform(X_sel)

            
            
            tree_transformer_inf = TreeTransformer(estimator = self.estimator, max_components= int(X_inf.shape[0]*max_components) )
            tree_transformer_inf.fit(X_inf) 
            transformed_feats_inf = tree_transformer_inf.transform(X_inf) 
            
            
            n_sel = len(y_sel)
            n_inf = len(y_inf)
            p_sel = transformed_feats_sel.shape[1]
            p_inf = transformed_feats_inf.shape[1]
            

            
            for j in range(X.shape[1]):
                stumps_sel_for_feat = tree_transformer_sel.original_feat_to_transformed_mapping[j]
                num_splits_for_feat = len(stumps_sel_for_feat)
                if num_splits_for_feat == 0:
                    p_vals[i,j] = 1.0
                else:
                    stumps_inf_for_feat = tree_transformer_inf.original_feat_to_transformed_mapping[j]
                    X_sel_for_feat = transformed_feats_sel[:,stumps_sel_for_feat]
                    p_sel_feat = X_sel_for_feat.shape[1]
                    sigma_sel = (np.sum((y_sel - np.mean(y_sel))**2))/(n_sel - p_sel_feat -1)
                    if eta is None:
                        eta = sigma_sel
                    X_inf_for_feat = transformed_feats_inf[:,stumps_inf_for_feat]
                    p_inf_feat = X_inf_for_feat.shape[1]
                    sigma_inf = (np.sum((y_inf - np.mean(y_inf))**2))/(n_inf - p_inf_feat -1)
                    optimal_lambda_for_feat = self.get_optimal_lambda(X_sel_for_feat,y_sel,eta,sigma_sel,lr,n_steps)
                    p_vals[i,j] = self.compute_p_val(optimal_lambda_for_feat,X_inf_for_feat,y_inf,sigma_inf,num_reps,n_sel,n_inf,p_sel_feat,p_inf_feat)
            
        median_p_vals = 2*np.median(p_vals,axis=0)
        median_p_vals[median_p_vals > 1.0] = 1.0

        return median_p_vals
    
    #self.multiple_testing_correction(median_p_vals)
    
    def multiple_testing_correction(self,p_vals,method = 'bonferroni',alpha = 0.05):
        return smt.multipletests(p_vals, method=method)[1] 
             
    def compute_p_val(self,optimal_lambda,X_inf,y_inf,sigma_inf,num_reps,n_sel,n_inf,p_sel_feat,p_inf_feat):
        u_inf, s_inf, vh_inf = np.linalg.svd(X_inf, full_matrices=False)
        optimal_weights = optimal_lambda#optimal_lambda.cpu().detach().numpy()
        weighted_chi_squared_samples = np.sort(np.array(self.get_weighted_chi_squared(optimal_weights,n_sel,n_inf,p_sel_feat,p_inf_feat,num_reps)))
        test_statistic = (np.sum((optimal_weights*(np.transpose(u_inf) @ y_inf))**2))/sigma_inf
        quantile = stats.percentileofscore(weighted_chi_squared_samples, test_statistic, 'rank') / 100
        return 1.0 - quantile
        
        
    def get_optimal_lambda(self,X,y,eta,sigma_sel,lr,n_steps):
        u_sel, s_sel, vh_sel = np.linalg.svd(X, full_matrices=False)
        betas = np.transpose(u_sel) @ y
        weights = []
        for i in range(u_sel.shape[1]):
            weights.append(np.random.uniform())
        weights = np.array(weights)
        
        #tns = torch.distributions.Uniform(0,1.0).sample((u_sel.shape[1],))
        #weights = Variable(tns, requires_grad=True)
        #opt = torch.optim.SGD([weights], lr=lr)
        for i in range(n_steps):
            gradient = self.compute_gradient(weights,betas,u_sel,y,eta,sigma_sel)
            weights = np.add(weights, lr*gradient)
            #print(weights)
            if all(gradient == 0.0): 
                break
            #opt.zero_grad()
            #z = self.forward(weights,u_sel,y,eta,sigma_sel)#torch.linalg.vector_norm(x)#x*x
            #z.sum().backward()
            #z.sum().backward()
            #print(weights.grad.data)
#            z.sum().backward() # Calculate gradients
            #opt.step()
        return weights
    
    def compute_gradient(self,weights,betas,u_sel,y_sel,eta,sigma_sel):
        gradients = []
        g = np.dot(weights**2,betas**2) - eta*LA.norm(weights)**2
        h = sigma_sel*np.sqrt(np.sum(weights**4))
        for i in range(len(weights)):
            g_prime = 2.0*weights[i]*(betas[i]**2) - (2.0*eta*weights[i])
            h_prime = ((2*(weights[i]**3))*(sigma_sel))/(np.sqrt(np.sum(weights**4)))
            grad_weight = (g_prime*h - h_prime*g)/(h**2)
            gradients.append(grad_weight)
        return np.array(gradients)
    
            
    def get_weighted_chi_squared(self,weights,n_sel,n_inf,p_sel_feat,p_inf_feat,num_reps = 10000):
        k = len(weights)
        samples = []
        for n in range(num_reps):
            numerator_sample = 0.0
            denominator_sample = np.random.chisquare(n_sel-p_inf_feat)
            for i in range(k):
                numerator_sample += (weights[i]**2)*np.random.chisquare(1, size=None)
            samples.append(numerator_sample/denominator_sample)
        return samples 
        
    
    
    
            
    #def forward(self,weights,u_sel,y_sel,eta,sigma_sel):
    #    T1 = torch.from_numpy(np.transpose(u_sel) @ y_sel)
    #    T1 = T1.type(torch.FloatTensor)
    #    T1 = weights * T1
    #    T1 = torch.linalg.vector_norm(T1)**2
    #    T2 = eta*torch.sum(weights**2)
     #   T3 = sigma_sel*torch.sqrt(torch.sum(weights**4))
     #   return torch.divide(torch.subtract(T2,T1),T3)
      #print("g_prime" + str(g_prime))
            #print("g" + str(g))
            #print("h_prime:" + str(h_prime))
            ##print("h" + str(h))
            #print("grad weight numerator:" + str(g_prime*h - h_prime*g))
            #print("grad_weight:" + str(grad_weight))
            #print(g_prime*h - h_prime*g[0])