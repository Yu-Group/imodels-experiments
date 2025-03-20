# sklearn imports
from sklearn.neighbors import NearestNeighbors, KNeighborsTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# data science imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# imodels imports
from imodels import get_clean_dataset
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusClassifier
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import RFPlusMDI

# data getters
from ucimlrepo import fetch_ucirepo

# local feature importance
import shap
import lime

# fetch dataset
breast_cancer_wisconsin_original = fetch_ucirepo(id=15)

# data
X = breast_cancer_wisconsin_original.data.features
y = np.array(breast_cancer_wisconsin_original.data.targets).flatten()

# remove rows with 'nan' entries for 'Bare_nuclei'
X = X.dropna()
# remove same observations from dataframe y
y = y[X.index]
# reset index
X = X.reset_index(drop=True)
X = X.to_numpy()

# transform y from 2/4 to 0/1
y = (y == 4).astype(int)

# center and scale the covariates
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split evenly into train/valid/test
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33,
                                                  random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.5,
                                                    random_state=42)

# fit random forest with params from MDI+
rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3,
                            max_features='sqrt', random_state=42)
rf.fit(X_train, y_train)

# elastic net rf+
rf_plus_elastic = RandomForestPlusClassifier(rf_model=rf,
            prediction_model=LogisticRegressionCV(penalty='elasticnet',
                    l1_ratios=[0.1,0.5,0.9,0.99], solver='saga', cv=3,
                n_jobs=-1, tol=5e-4, max_iter=5000, random_state=42))
rf_plus_elastic.fit(X_train, y_train)

def get_lime(X: np.ndarray, rf):
    """
    Get the LIME values and rankings for the given data.
    
    Inputs:
    - X (np.ndarray): The feature matrix.
    - rf (RandomForestClassifier/Regressor): The fitted RF object.
    
    Outputs:
    - lime_values (np.ndarray): The LIME values.
    """
    
    lime_values = np.zeros((X.shape[0], X.shape[1]))
    explainer = lime.lime_tabular.LimeTabularExplainer(X, verbose = False,
                                                       mode = "classification")
    num_features = X.shape[1]
    for i in range(X.shape[0]):
        exp = explainer.explain_instance(X[i, :], rf.predict_proba,
                                         num_features = num_features)
        original_feature_importance = exp.as_map()[1]
        sorted_feature_importance = sorted(original_feature_importance, key=lambda x: x[0])
        for j in range(num_features):
            lime_values[i, j] = sorted_feature_importance[j][1]
        
    return lime_values

# get shap
shap_explainer = shap.TreeExplainer(rf)
shap_valid = shap_explainer.shap_values(X_valid,
                                        check_additivity = False)[:, :, 1]
shap_test = shap_explainer.shap_values(X_test,
                                        check_additivity = False)[:, :, 1]

# get lime
lime_valid = get_lime(X_valid, rf)
lime_test = get_lime(X_test, rf)

# get lmdi values
mdi_explainer = RFPlusMDI(rf_plus_elastic, mode = "only_k", evaluate_on = 'all')
lmdi_valid = mdi_explainer.explain_linear_partial(X_valid, y_valid,
                                                  normalize=False,
                                                  square=False, ranking=False)
lmdi_test = mdi_explainer.explain_linear_partial(X_test, y_test,
                                                    normalize=False,
                                                    square=False, ranking=False)

# get all neighbors of each point
shap_nbrs = NearestNeighbors(n_neighbors=len(X_valid))
shap_nbrs.fit(shap_valid)
shap_dist, shap_idxs = shap_nbrs.kneighbors(shap_test)
lime_nbrs = NearestNeighbors(n_neighbors=len(X_valid))
lime_nbrs.fit(lime_valid)
lime_dist, lime_idxs = lime_nbrs.kneighbors(lime_test)
lmdi_nbrs = NearestNeighbors(n_neighbors=len(X_valid))
lmdi_nbrs.fit(lmdi_valid)
lmdi_dist, lmdi_idxs = lmdi_nbrs.kneighbors(lmdi_test)

# find the three closest neighbors to each point that have the opposite label
shap_opposite = []
for i in range(len(y_test)):
    if y_test[i] == 1:
        opposite = np.where(y_valid == 0)[0]
    else:
        opposite = np.where(y_valid == 1)[0]
    distances = shap_dist[i][np.isin(shap_idxs[i], opposite)]
    closest = np.argsort(distances)[:3]
    shap_opposite.append(shap_idxs[i][np.isin(shap_idxs[i], opposite)][closest])
shap_opposite = np.array(shap_opposite)
lime_opposite = []
for i in range(len(y_test)):
    if y_test[i] == 1:
        opposite = np.where(y_valid == 0)[0]
    else:
        opposite = np.where(y_valid == 1)[0]
    distances = lime_dist[i][np.isin(lime_idxs[i], opposite)]
    closest = np.argsort(distances)[:3]
    lime_opposite.append(lime_idxs[i][np.isin(lime_idxs[i], opposite)][closest])
lime_opposite = np.array(lime_opposite)
lmdi_opposite = []
for i in range(len(y_test)):
    if y_test[i] == 1:
        opposite = np.where(y_valid == 0)[0]
    else:
        opposite = np.where(y_valid == 1)[0]
    distances = lmdi_dist[i][np.isin(lmdi_idxs[i], opposite)]
    closest = np.argsort(distances)[:3]
    lmdi_opposite.append(lmdi_idxs[i][np.isin(lmdi_idxs[i], opposite)][closest])
lmdi_opposite = np.array(lmdi_opposite)

shap_distances = []
for i in range(len(y_test)):
    distances = []
    for j in range(3):
        distances.append(np.linalg.norm(X_test[i] - X_valid[shap_opposite[i][j]]))
    shap_distances.append(distances)
shap_distances = np.array(shap_distances)
shap_distances = shap_distances.mean(axis=1)
lime_distances = []
for i in range(len(y_test)):
    distances = []
    for j in range(3):
        distances.append(np.linalg.norm(X_test[i] - X_valid[lime_opposite[i][j]]))
    lime_distances.append(distances)
lime_distances = np.array(lime_distances)
lime_distances = lime_distances.mean(axis=1)
lmdi_distances = []
for i in range(len(y_test)):
    distances = []
    for j in range(3):
        distances.append(np.linalg.norm(X_test[i] - X_valid[lmdi_opposite[i][j]]))
    lmdi_distances.append(distances)
lmdi_distances = np.array(lmdi_distances)
lmdi_distances = lmdi_distances.mean(axis=1)

# plot distances
plt.figure(figsize=(10, 6))
plt.hist(shap_distances, bins=30, alpha=0.5, label='SHAP', color='blue')
plt.hist(lime_distances, bins=30, alpha=0.5, label='LIME', color='orange')
plt.hist(lmdi_distances, bins=30, alpha=0.5, label='LMDI', color='green')
plt.xlabel('Average Distance to 3 Closest Opposite Label Neighbors')
plt.ylabel('Frequency')
plt.title('Distribution of Distances to Three Closest Opposite Label Neighbors')
plt.legend()
plt.show()