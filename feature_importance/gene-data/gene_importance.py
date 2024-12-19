import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusRegressor
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import RFPlusMDI

if __name__ == '__main__':
    
    # read in data
    genotype = pd.read_csv("/scratch/users/omer_ronen/mutemb_esm/X_k_5_ilvm_esm_prod_pppl_full_ptv.csv")
    # genotype = pd.read_csv("/scratch/users/omer_ronen/mutemb_esm/X_k_5_ilvm_oh.csv")
    phenotype = pd.read_csv('/scratch/users/omer_ronen/mutemb_esm/y_ilvm_oh.csv')
    # make genotype a numpy array
    genotype = genotype.to_numpy()
    # make phenotype a 1D numpy array
    phenotype = phenotype.to_numpy().reshape(-1)
    
    # train-test split
    genotype_train, genotype_test, phenotype_train, phenotype_test = \
        train_test_split(genotype, phenotype, test_size = 0.3,
                         random_state = 1)
        
    print("Data split")
        
    # fit random forest model
    rf = RandomForestRegressor(n_estimators = 100, max_depth=5, random_state = 42)
    rf.fit(genotype_train, phenotype_train)
    
    print("RF fitted")
    
    # predict on test set
    phenotype_pred = rf.predict(genotype_test)
    
    print("Predictions obtained")
    
    # calculate correlation for pred-check
    cor = np.corrcoef(phenotype_test, phenotype_pred)[0,1]
    
    print("Correlation calculated: ", cor)
    
    # get MDI feature importance scores
    importances = rf.feature_importances_
    
    print("Importance scores obtained")
    
    # write importances to a file
    np.savetxt('mdi_importance_scores_subset_embedding_data.csv', importances, delimiter = ',')
    
    print("Importance scores written to mdi_importance_scores_subset_embedding_data.csv")
    
    # subset the genotype data to only include the top 1000 features
    top_1000_features = np.argsort(importances)[::-1][:1000]
    genotype_train_subset = genotype_train[:, top_1000_features]
    genotype_test_subset = genotype_test[:, top_1000_features]
    
    # fit new rf on subset
    rf_subset = RandomForestRegressor(n_estimators = 100, max_depth=5, random_state = 42)
    rf_subset.fit(genotype_train_subset, phenotype_train)
    
    print("Subset RF fitted")
    
    # predict on test set
    phenotype_subset_pred = rf_subset.predict(genotype_test_subset)
    
    print("Subset predictions obtained")
    
    # calculate correlation for pred-check
    subset_cor = np.corrcoef(phenotype_test, phenotype_subset_pred)[0,1]
    
    print("Subset correlation calculated: ", subset_cor)
    
    # get MDI feature importance scores
    subset_importances = rf_subset.feature_importances_
    
    print("Subset importance scores obtained")
    
    # write importances to a file
    np.savetxt('mdi_importance_scores_subset_rf_embedding_data.csv', importances, delimiter = ',')
    
    print("Importance scores written to mdi_importance_scores_subset_rf_embedding_data.csv")
    
    # fit rf+
    rf_plus = RandomForestPlusRegressor(rf_model=rf_subset, prediction_model=Ridge())
    rf_plus.fit(genotype_train_subset, phenotype_train)
    
    # get lmdi+
    lmdi_explainer = RFPlusMDI(rf_plus)
    lmdi_importances = lmdi_explainer.explain_linear_partial(X=genotype_train_subset, y=phenotype_train, l2norm=True, sign=True, normalize=True, njobs=-1)
    
    # write lmdi+ importances to a file
    np.savetxt('lmdi_plus_importance_scores_subset_embedding_data.csv', lmdi_importances, delimiter = ',')
    
    print("Importance scores written to lmdi_plus_importance_scores_subset_embedding_data.csv")
    