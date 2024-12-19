# import required packages
from imodels import get_clean_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusClassifier
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import  RFPlusMDI
from subgroup_detection import *
import warnings
import shap
warnings.filterwarnings('ignore', category=DeprecationWarning)

print("line 16")

def load_data():
    print("began load_data")
    # get pre-cleaned compas dataset from imodels
    X, y, feature_names = get_clean_dataset('compas_two_year_clean', data_source='imodels')
    X = pd.DataFrame(X, columns=feature_names)
    
    # the propublica study narrowed the dataset to only African-American and
    # Caucasian defendants, and doing so keeps the vast majority of the data,
    # so we will do the same.
    y = y[(X['race:African-American'] == 1) | (X['race:Caucasian'] == 1)]
    X = X[(X['race:African-American'] == 1) | (X['race:Caucasian'] == 1)]

    # now that we have narrowed the dataset, we should remove the one-hot encodings
    # of variables that are consistently zero, such as the other ethnicities.
    # we also drop age because the binned 'age category' is preferred here.
    X = X.drop(["race:Asian", "race:Hispanic", "race:Native_American",
                "race:Other", "age"], axis = 1)
    
    # we dont want y as a pandas series
    y = np.asarray(y)
    print("data loaded")
    return X, y

def split_data(X, y, random_state):
    print("began split_data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=random_state)
    print("data split")
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    print("began train_models")
    log = LogisticRegression(random_state=0, max_iter=1000)
    log.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    rf_plus = RandomForestPlusClassifier(rf)
    rf_plus.fit(X_train, y_train)
    print("models trained")
    return log, rf, rf_plus

def lmdi_plus(X_train, y_train, X_test, y_test, log, rf, rf_plus):
    print("began lmdi_plus")
    # get feature importances
    mdi_explainer = RFPlusMDI(rf_plus, prediction_model = LogisticRegression())
    mdi, partial_preds = mdi_explainer.explain(np.asarray(X_train), y_train)
    mdi_rankings = mdi_explainer.get_rankings(mdi)
    
    # get rbo distance matrix
    rbo_train = compute_rbo_matrix(mdi_rankings, form = 'distance')
    
    mdi_copy = pd.DataFrame(mdi, columns=X_train.columns).copy()
    num_clusters = 4
    clusters = assign_training_clusters(mdi_copy, rbo_train, num_clusters)
    
    # get mdi rankings assignments for test points
    mdi_test, partial_preds_test = mdi_explainer.explain(np.asarray(X_test))
    mdi_test_rankings = mdi_explainer.get_rankings(mdi_test)
    
    test_clust = assign_testing_clusters(method = "centroid", median_approx = False,
                                     rbo_distance_matrix = rbo_train,
                                     lfi_train_ranking = mdi_rankings,
                                     lfi_test_ranking = mdi_test_rankings,
                                     clusters = clusters)
    print("testing clusters assigned")
    
    cluster1_trainX = X_train[clusters == 1]
    cluster2_trainX = X_train[clusters == 2]
    cluster3_trainX = X_train[clusters == 3]
    cluster4_trainX = X_train[clusters == 4]

    cluster1_trainy = y_train[clusters == 1]
    cluster2_trainy = y_train[clusters == 2]
    cluster3_trainy = y_train[clusters == 3]
    cluster4_trainy = y_train[clusters == 4]

    cluster1_testX = X_test[test_clust == 1]
    cluster2_testX = X_test[test_clust == 2]
    cluster3_testX = X_test[test_clust == 3]
    cluster4_testX = X_test[test_clust == 4]

    cluster1_testy = y_test[test_clust == 1]
    cluster2_testy = y_test[test_clust == 2]
    cluster3_testy = y_test[test_clust == 3]
    cluster4_testy = y_test[test_clust == 4]
    
    # fit RF+ on each training set, predict test
    rf1 = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_plus1 = RandomForestPlusClassifier(rf1)
    rf_plus1.fit(cluster1_trainX, cluster1_trainy)

    rf2 = RandomForestClassifier(n_estimators=100, random_state=1)
    rf_plus2 = RandomForestPlusClassifier(rf2)
    rf_plus2.fit(cluster2_trainX, cluster2_trainy)

    rf3 = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_plus3 = RandomForestPlusClassifier(rf3)
    rf_plus3.fit(cluster3_trainX, cluster3_trainy)

    rf4 = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_plus4 = RandomForestPlusClassifier(rf4)
    rf_plus4.fit(cluster4_trainX, cluster4_trainy)
    
    # fit RF on each training set, predict test
    rf1.fit(cluster1_trainX, cluster1_trainy)

    rf2.fit(cluster2_trainX, cluster2_trainy)

    rf3.fit(cluster3_trainX, cluster3_trainy)

    rf4.fit(cluster4_trainX, cluster4_trainy)
    
    # fit log model on each training set, predict test
    log1 = LogisticRegression(random_state=0, max_iter=1000)
    log1.fit(cluster1_trainX, cluster1_trainy)

    log2 = LogisticRegression(random_state=0, max_iter=1000)
    log2.fit(cluster2_trainX, cluster2_trainy)
    
    log3 = LogisticRegression(random_state=0, max_iter=1000)
    log3.fit(cluster3_trainX, cluster3_trainy)
    
    log4 = LogisticRegression(random_state=0, max_iter=1000)
    log4.fit(cluster4_trainX, cluster4_trainy)
    print("local models fit")
    
    cols = ['global_log_accuracy', 'global_log_misclassified',
            'global_log_auroc', 'global_log_auprc', 'global_log_f1',
            'global_rf_accuracy', 'global_rf_misclassified', 'global_rf_auroc',
            'global_rf_auprc', 'global_rf_f1', 'global_rf_plus_accuracy',
            'global_rf_plus_misclassified', 'global_rf_plus_auroc',
            'global_rf_plus_auprc', 'global_rf_plus_f1', 'local_log_accuracy',
            'local_log_misclassified', 'local_log_auroc', 'local_log_auprc',
            'local_log_f1', 'local_rf_accuracy', 'local_rf_misclassified',
            'local_rf_auroc', 'local_rf_auprc', 'local_rf_f1',
            'local_rf_plus_accuracy', 'local_rf_plus_misclassified',
            'local_rf_plus_auroc', 'local_rf_plus_auprc', 'local_rf_plus_f1']
    
    # c1_results = pd.DataFrame(columns=cols)
    # c2_results = pd.DataFrame(columns=cols)
    # c3_results = pd.DataFrame(columns=cols)
    # c4_results = pd.DataFrame(columns=cols)
    
    local_log_preds1 = log1.predict(cluster1_testX)
    local_log_acc1 = np.mean(cluster1_testy == local_log_preds1)
    local_log_mis1 = np.sum(cluster1_testy != local_log_preds1)
    local_log_auroc1 = roc_auc_score(cluster1_testy,
                                    log1.predict_proba(cluster1_testX)[:, 1])
    local_log_auprc1 = average_precision_score(cluster1_testy,
                                    log1.predict_proba(cluster1_testX)[:, 1])
    local_log_f1score1 = f1_score(cluster1_testy, local_log_preds1)
    
    global_log_preds1 = log.predict(cluster1_testX)
    global_log_acc1 = np.mean(cluster1_testy == global_log_preds1)
    global_log_mis1 = np.sum(cluster1_testy != global_log_preds1)
    global_log_auroc1 = roc_auc_score(cluster1_testy,
                                    log.predict_proba(cluster1_testX)[:, 1])
    global_log_auprc1 = average_precision_score(cluster1_testy,
                                    log.predict_proba(cluster1_testX)[:, 1])
    global_log_f1score1 = f1_score(cluster1_testy, global_log_preds1)
    
    local_rf_preds1 = rf1.predict(cluster1_testX)
    local_rf_acc1 = np.mean(cluster1_testy == local_rf_preds1)
    local_rf_mis1 = np.sum(cluster1_testy != local_rf_preds1)
    local_rf_auroc1 = roc_auc_score(cluster1_testy,
                                    rf1.predict_proba(cluster1_testX)[:, 1])
    local_rf_auprc1 = average_precision_score(cluster1_testy,
                                    rf1.predict_proba(cluster1_testX)[:, 1])
    local_rf_f1score1 = f1_score(cluster1_testy, local_rf_preds1)
    
    global_rf_preds1 = rf.predict(cluster1_testX)
    global_rf_acc1 = np.mean(cluster1_testy == global_rf_preds1)
    global_rf_mis1 = np.sum(cluster1_testy != global_rf_preds1)
    global_rf_auroc1 = roc_auc_score(cluster1_testy,
                                    rf.predict_proba(cluster1_testX)[:, 1])
    global_rf_auprc1 = average_precision_score(cluster1_testy,
                                    rf.predict_proba(cluster1_testX)[:, 1])
    global_rf_f1score1 = f1_score(cluster1_testy, global_rf_preds1)
    
    local_rf_plus_probpreds1 = rf_plus1.predict_proba(cluster1_testX)[:, 1]
    local_rf_plus_preds1 = local_rf_plus_probpreds1 > 0.5
    local_rf_plus_acc1 = np.mean(cluster1_testy == local_rf_plus_preds1)
    local_rf_plus_mis1 = np.sum(cluster1_testy != local_rf_plus_preds1)
    local_rf_plus_auroc1 = roc_auc_score(cluster1_testy,
                                         local_rf_plus_probpreds1)
    local_rf_plus_auprc1 = average_precision_score(cluster1_testy,
                                                   local_rf_plus_probpreds1)
    local_rf_plus_f1score1 = f1_score(cluster1_testy, local_rf_plus_preds1)
    
    global_rf_plus_probpreds1 = rf_plus.predict_proba(cluster1_testX)[:, 1]
    global_rf_plus_preds1 = global_rf_plus_probpreds1 > 0.5
    global_rf_plus_acc1 = np.mean(cluster1_testy == global_rf_plus_preds1)
    global_rf_plus_mis1 = np.sum(cluster1_testy != global_rf_plus_preds1)
    global_rf_plus_auroc1 = roc_auc_score(cluster1_testy,
                                            global_rf_plus_probpreds1)
    global_rf_plus_auprc1 = average_precision_score(cluster1_testy,
                                            global_rf_plus_probpreds1)
    global_rf_plus_f1score1 = f1_score(cluster1_testy, global_rf_plus_preds1)
    
    # add row to cluster 1 results data
    c1_results = np.asarray([global_log_acc1, global_log_mis1, global_log_auroc1,
                        global_log_auprc1, global_log_f1score1, global_rf_acc1,
                        global_rf_mis1, global_rf_auroc1, global_rf_auprc1,
                        global_rf_f1score1, global_rf_plus_acc1, global_rf_plus_mis1,
                        global_rf_plus_auroc1, global_rf_plus_auprc1,
                        global_rf_plus_f1score1,                    
                        local_log_acc1, local_log_mis1,
                        local_log_auroc1, local_log_auprc1, local_log_f1score1,
                        local_rf_acc1, local_rf_mis1, local_rf_auroc1,
                        local_rf_auprc1, local_rf_f1score1, local_rf_plus_acc1,
                        local_rf_plus_mis1, local_rf_plus_auroc1,
                        local_rf_plus_auprc1, local_rf_plus_f1score1])
    print("cluster 1 results complete")
    
    local_log_preds2 = log2.predict(cluster2_testX)
    local_log_acc2 = np.mean(cluster2_testy == local_log_preds2)
    local_log_mis2 = np.sum(cluster2_testy != local_log_preds2)
    local_log_auroc2 = roc_auc_score(cluster2_testy,
                                    log2.predict_proba(cluster2_testX)[:, 1])
    local_log_auprc2 = average_precision_score(cluster2_testy,
                                    log2.predict_proba(cluster2_testX)[:, 1])
    local_log_f1score2 = f1_score(cluster2_testy, local_log_preds2)
    
    global_log_preds2 = log.predict(cluster2_testX)
    global_log_acc2 = np.mean(cluster2_testy == global_log_preds2)
    global_log_mis2 = np.sum(cluster2_testy != global_log_preds2)
    global_log_auroc2 = roc_auc_score(cluster2_testy,
                                    log.predict_proba(cluster2_testX)[:, 1])
    global_log_auprc2 = average_precision_score(cluster2_testy,
                                    log.predict_proba(cluster2_testX)[:, 1])
    global_log_f1score2 = f1_score(cluster2_testy, global_log_preds2)
    
    local_rf_preds2 = rf2.predict(cluster2_testX)
    local_rf_acc2 = np.mean(cluster2_testy == local_rf_preds2)
    local_rf_mis2 = np.sum(cluster2_testy != local_rf_preds2)
    local_rf_auroc2 = roc_auc_score(cluster2_testy,
                                    rf2.predict_proba(cluster2_testX)[:, 1])
    local_rf_auprc2 = average_precision_score(cluster2_testy,
                                    rf2.predict_proba(cluster2_testX)[:, 1])
    local_rf_f1score2 = f1_score(cluster2_testy, local_rf_preds2)
    
    global_rf_preds2 = rf.predict(cluster2_testX)
    global_rf_acc2 = np.mean(cluster2_testy == global_rf_preds2)
    global_rf_mis2 = np.sum(cluster2_testy != global_rf_preds2)
    global_rf_auroc2 = roc_auc_score(cluster2_testy,
                                    rf.predict_proba(cluster2_testX)[:, 1])
    global_rf_auprc2 = average_precision_score(cluster2_testy,
                                    rf.predict_proba(cluster2_testX)[:, 1])
    global_rf_f1score2 = f1_score(cluster2_testy, global_rf_preds2)
    
    local_rf_plus_probpreds2 = rf_plus2.predict_proba(cluster2_testX)[:, 1]
    local_rf_plus_preds2 = local_rf_plus_probpreds2 > 0.5
    local_rf_plus_acc2 = np.mean(cluster2_testy == local_rf_plus_preds2)
    local_rf_plus_mis2 = np.sum(cluster2_testy != local_rf_plus_preds2)
    local_rf_plus_auroc2 = roc_auc_score(cluster2_testy,
                                         local_rf_plus_probpreds2)
    local_rf_plus_auprc2 = average_precision_score(cluster2_testy,
                                                   local_rf_plus_probpreds2)
    local_rf_plus_f1score2 = f1_score(cluster2_testy, local_rf_plus_preds2)
    
    global_rf_plus_probpreds2 = rf_plus.predict_proba(cluster2_testX)[:, 1]
    global_rf_plus_preds2 = global_rf_plus_probpreds2 > 0.5
    global_rf_plus_acc2 = np.mean(cluster2_testy == global_rf_plus_preds2)
    global_rf_plus_mis2 = np.sum(cluster2_testy != global_rf_plus_preds2)
    global_rf_plus_auroc2 = roc_auc_score(cluster2_testy,
                                            global_rf_plus_probpreds2)
    global_rf_plus_auprc2 = average_precision_score(cluster2_testy,
                                            global_rf_plus_probpreds2)
    global_rf_plus_f1score2 = f1_score(cluster2_testy, global_rf_plus_preds2)
    
    # add row to cluster 2 results data
    c2_results = np.asarray([global_log_acc2, global_log_mis2, global_log_auroc2,
                        global_log_auprc2, global_log_f1score2, global_rf_acc2,
                        global_rf_mis2, global_rf_auroc2, global_rf_auprc2,
                        global_rf_f1score2, global_rf_plus_acc2, global_rf_plus_mis2,
                        global_rf_plus_auroc2, global_rf_plus_auprc2,
                        global_rf_plus_f1score2, 
                        local_log_acc2, local_log_mis2,
                        local_log_auroc2, local_log_auprc2, local_log_f1score2,
                        local_rf_acc2, local_rf_mis2, local_rf_auroc2,
                        local_rf_auprc2, local_rf_f1score2, local_rf_plus_acc2,
                        local_rf_plus_mis2, local_rf_plus_auroc2,
                        local_rf_plus_auprc2, local_rf_plus_f1score2])
    print("cluster 2 results complete")
    
    local_log_preds3 = log3.predict(cluster3_testX)
    local_log_acc3 = np.mean(cluster3_testy == local_log_preds3)
    local_log_mis3 = np.sum(cluster3_testy != local_log_preds3)
    local_log_auroc3 = roc_auc_score(cluster3_testy,
                                    log3.predict_proba(cluster3_testX)[:, 1])
    local_log_auprc3 = average_precision_score(cluster3_testy,
                                    log3.predict_proba(cluster3_testX)[:, 1])
    local_log_f1score3 = f1_score(cluster3_testy, local_log_preds3)
    
    global_log_preds3 = log.predict(cluster3_testX)
    global_log_acc3 = np.mean(cluster3_testy == global_log_preds3)
    global_log_mis3 = np.sum(cluster3_testy != global_log_preds3)
    global_log_auroc3 = roc_auc_score(cluster3_testy,
                                    log.predict_proba(cluster3_testX)[:, 1])
    global_log_auprc3 = average_precision_score(cluster3_testy,
                                    log.predict_proba(cluster3_testX)[:, 1])
    global_log_f1score3 = f1_score(cluster3_testy, global_log_preds3)
    
    local_rf_preds3 = rf3.predict(cluster3_testX)
    local_rf_acc3 = np.mean(cluster3_testy == local_rf_preds3)
    local_rf_mis3 = np.sum(cluster3_testy != local_rf_preds3)
    local_rf_auroc3 = roc_auc_score(cluster3_testy,
                                    rf3.predict_proba(cluster3_testX)[:, 1])
    local_rf_auprc3 = average_precision_score(cluster3_testy,
                                    rf3.predict_proba(cluster3_testX)[:, 1])
    local_rf_f1score3 = f1_score(cluster3_testy, local_rf_preds3)
    
    global_rf_preds3 = rf.predict(cluster3_testX)
    global_rf_acc3 = np.mean(cluster3_testy == global_rf_preds3)
    global_rf_mis3 = np.sum(cluster3_testy != global_rf_preds3)
    global_rf_auroc3 = roc_auc_score(cluster3_testy,
                                    rf.predict_proba(cluster3_testX)[:, 1])
    global_rf_auprc3 = average_precision_score(cluster3_testy,
                                    rf.predict_proba(cluster3_testX)[:, 1])
    global_rf_f1score3 = f1_score(cluster3_testy, global_rf_preds3)
    
    local_rf_plus_probpreds3 = rf_plus3.predict_proba(cluster3_testX)[:, 1]
    local_rf_plus_preds3 = local_rf_plus_probpreds3 > 0.5
    local_rf_plus_acc3 = np.mean(cluster3_testy == local_rf_plus_preds3)
    local_rf_plus_mis3 = np.sum(cluster3_testy != local_rf_plus_preds3)
    local_rf_plus_auroc3 = roc_auc_score(cluster3_testy,
                                         local_rf_plus_probpreds3)
    local_rf_plus_auprc3 = average_precision_score(cluster3_testy,
                                                   local_rf_plus_probpreds3)
    local_rf_plus_f1score3 = f1_score(cluster3_testy, local_rf_plus_preds3)
    
    global_rf_plus_probpreds3 = rf_plus.predict_proba(cluster3_testX)[:, 1]
    global_rf_plus_preds3 = global_rf_plus_probpreds3 > 0.5
    global_rf_plus_acc3 = np.mean(cluster3_testy == global_rf_plus_preds3)
    global_rf_plus_mis3 = np.sum(cluster3_testy != global_rf_plus_preds3)
    global_rf_plus_auroc3 = roc_auc_score(cluster3_testy,
                                            global_rf_plus_probpreds3)
    global_rf_plus_auprc3 = average_precision_score(cluster3_testy,
                                            global_rf_plus_probpreds3)
    global_rf_plus_f1score3 = f1_score(cluster3_testy, global_rf_plus_preds3)
    
    # add row to cluster 3 results data
    c3_results = np.asarray([global_log_acc3, global_log_mis3, global_log_auroc3,
                        global_log_auprc3, global_log_f1score3, global_rf_acc3,
                        global_rf_mis3, global_rf_auroc3, global_rf_auprc3,
                        global_rf_f1score3, global_rf_plus_acc3, global_rf_plus_mis3,
                        global_rf_plus_auroc3, global_rf_plus_auprc3,
                        global_rf_plus_f1score3,
                        local_log_acc3, local_log_mis3,
                        local_log_auroc3, local_log_auprc3, local_log_f1score3,
                        local_rf_acc3, local_rf_mis3, local_rf_auroc3,
                        local_rf_auprc3, local_rf_f1score3, local_rf_plus_acc3,
                        local_rf_plus_mis3, local_rf_plus_auroc3,
                        local_rf_plus_auprc3, local_rf_plus_f1score3])
    
    print("cluster 3 results complete")
    
    local_log_preds4 = log4.predict(cluster4_testX)
    local_log_acc4 = np.mean(cluster4_testy == local_log_preds4)
    local_log_mis4 = np.sum(cluster4_testy != local_log_preds4)
    local_log_auroc4 = roc_auc_score(cluster4_testy,
                                    log4.predict_proba(cluster4_testX)[:, 1])
    local_log_auprc4 = average_precision_score(cluster4_testy,
                                    log4.predict_proba(cluster4_testX)[:, 1])
    local_log_f1score4 = f1_score(cluster4_testy, local_log_preds4)
    
    global_log_preds4 = log.predict(cluster4_testX)
    global_log_acc4 = np.mean(cluster4_testy == global_log_preds4)
    global_log_mis4 = np.sum(cluster4_testy != global_log_preds4)
    global_log_auroc4 = roc_auc_score(cluster4_testy,
                                    log.predict_proba(cluster4_testX)[:, 1])
    global_log_auprc4 = average_precision_score(cluster4_testy,
                                    log.predict_proba(cluster4_testX)[:, 1])
    global_log_f1score4 = f1_score(cluster4_testy, global_log_preds4)
    
    local_rf_preds4 = rf4.predict(cluster4_testX)
    local_rf_acc4 = np.mean(cluster4_testy == local_rf_preds4)
    local_rf_mis4 = np.sum(cluster4_testy != local_rf_preds4)
    local_rf_auroc4 = roc_auc_score(cluster4_testy,
                                    rf4.predict_proba(cluster4_testX)[:, 1])
    local_rf_auprc4 = average_precision_score(cluster4_testy,
                                    rf4.predict_proba(cluster4_testX)[:, 1])
    local_rf_f1score4 = f1_score(cluster4_testy, local_rf_preds4)
    
    global_rf_preds4 = rf.predict(cluster4_testX)
    global_rf_acc4 = np.mean(cluster4_testy == global_rf_preds4)
    global_rf_mis4 = np.sum(cluster4_testy != global_rf_preds4)
    global_rf_auroc4 = roc_auc_score(cluster4_testy,
                                    rf.predict_proba(cluster4_testX)[:, 1])
    global_rf_auprc4 = average_precision_score(cluster4_testy,
                                    rf.predict_proba(cluster4_testX)[:, 1])
    global_rf_f1score4 = f1_score(cluster4_testy, global_rf_preds4)
    
    local_rf_plus_probpreds4 = rf_plus4.predict_proba(cluster4_testX)[:, 1]
    local_rf_plus_preds4 = local_rf_plus_probpreds4 > 0.5
    local_rf_plus_acc4 = np.mean(cluster4_testy == local_rf_plus_preds4)
    local_rf_plus_mis4 = np.sum(cluster4_testy != local_rf_plus_preds4)
    local_rf_plus_auroc4 = roc_auc_score(cluster4_testy,
                                         local_rf_plus_probpreds4)
    local_rf_plus_auprc4 = average_precision_score(cluster4_testy,
                                                   local_rf_plus_probpreds4)
    local_rf_plus_f1score4 = f1_score(cluster4_testy, local_rf_plus_preds4)
    
    global_rf_plus_probpreds4 = rf_plus.predict_proba(cluster4_testX)[:, 1]
    global_rf_plus_preds4 = global_rf_plus_probpreds4 > 0.5
    global_rf_plus_acc4 = np.mean(cluster4_testy == global_rf_plus_preds4)
    global_rf_plus_mis4 = np.sum(cluster4_testy != global_rf_plus_preds4)
    global_rf_plus_auroc4 = roc_auc_score(cluster4_testy,
                                            global_rf_plus_probpreds4)
    global_rf_plus_auprc4 = average_precision_score(cluster4_testy,
                                            global_rf_plus_probpreds4)
    global_rf_plus_f1score4 = f1_score(cluster4_testy, global_rf_plus_preds4)
    
    # add row to cluster 4 results data
    c4_results = np.asarray([global_log_acc4, global_log_mis4, global_log_auroc4,
                        global_log_auprc4, global_log_f1score4, global_rf_acc4,
                        global_rf_mis4, global_rf_auroc4, global_rf_auprc4,
                        global_rf_f1score4, global_rf_plus_acc4, global_rf_plus_mis4,
                        global_rf_plus_auroc4, global_rf_plus_auprc4,
                        global_rf_plus_f1score4,
                        local_log_acc4, local_log_mis4,
                        local_log_auroc4, local_log_auprc4, local_log_f1score4,
                        local_rf_acc4, local_rf_mis4, local_rf_auroc4,
                        local_rf_auprc4, local_rf_f1score4, local_rf_plus_acc4,
                        local_rf_plus_mis4, local_rf_plus_auroc4,
                        local_rf_plus_auprc4, local_rf_plus_f1score4])
    
    print("cluster 4 results complete")
    
    # combine cluster results into one array where the clusters are the rows
    result = np.vstack((c1_results, c2_results, c3_results, c4_results))
    
    # Convert the NumPy array to a pandas DataFrame
    cluster_results = pd.DataFrame(result)

    # Set the column names
    cluster_results.columns = cols

    
    # row names
    cluster_results.index = ['cluster1', 'cluster2', 'cluster3', 'cluster4']
    return cluster_results

def tree_shap(X_train, y_train, X_test, y_test, log, rf, rf_plus):
    print("began tree_shap")
    # get feature importances
    shap_explainer = shap.TreeExplainer(rf)
    shap_values = np.abs(shap_explainer.shap_values(X_train, check_additivity=False))
    mdi_explainer = AloRFPlusMDI(rf_plus, evaluate_on='oob')
    shap_rankings = mdi_explainer.get_rankings(shap_values)[:,:,0]
        
    # get rbo distance matrix
    rbo_train = compute_rbo_matrix(shap_rankings, form = 'distance')
    
    shap_copy = pd.DataFrame(shap_values[:,:,0], columns=X_train.columns).copy()
    num_clusters = 4
    clusters = assign_training_clusters(shap_copy, rbo_train, num_clusters)
    
    # get mdi rankings assignments for test points
    shap_test_values = np.abs(shap_explainer.shap_values(X_test, check_additivity=False))
    shap_test_rankings = mdi_explainer.get_rankings(shap_test_values)[:,:,0]
    
    test_clust = assign_testing_clusters(method = "centroid", median_approx = False,
                                     rbo_distance_matrix = rbo_train,
                                     lfi_train_ranking = shap_rankings,
                                     lfi_test_ranking = shap_test_rankings,
                                     clusters = clusters)
    print("testing clusters assigned")
    
    cluster1_trainX = X_train[clusters == 1]
    cluster2_trainX = X_train[clusters == 2]
    cluster3_trainX = X_train[clusters == 3]
    cluster4_trainX = X_train[clusters == 4]

    cluster1_trainy = y_train[clusters == 1]
    cluster2_trainy = y_train[clusters == 2]
    cluster3_trainy = y_train[clusters == 3]
    cluster4_trainy = y_train[clusters == 4]

    cluster1_testX = X_test[test_clust == 1]
    cluster2_testX = X_test[test_clust == 2]
    cluster3_testX = X_test[test_clust == 3]
    cluster4_testX = X_test[test_clust == 4]

    cluster1_testy = y_test[test_clust == 1]
    cluster2_testy = y_test[test_clust == 2]
    cluster3_testy = y_test[test_clust == 3]
    cluster4_testy = y_test[test_clust == 4]
    
    # fit RF+ on each training set, predict test
    rf1 = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_plus1 = RandomForestPlusClassifier(rf1)
    rf_plus1.fit(cluster1_trainX, cluster1_trainy)

    rf2 = RandomForestClassifier(n_estimators=100, random_state=1)
    rf_plus2 = RandomForestPlusClassifier(rf2)
    rf_plus2.fit(cluster2_trainX, cluster2_trainy)

    rf3 = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_plus3 = RandomForestPlusClassifier(rf3)
    rf_plus3.fit(cluster3_trainX, cluster3_trainy)

    rf4 = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_plus4 = RandomForestPlusClassifier(rf4)
    rf_plus4.fit(cluster4_trainX, cluster4_trainy)
    
    # fit RF on each training set, predict test
    rf1.fit(cluster1_trainX, cluster1_trainy)

    rf2.fit(cluster2_trainX, cluster2_trainy)

    rf3.fit(cluster3_trainX, cluster3_trainy)

    rf4.fit(cluster4_trainX, cluster4_trainy)
    
    # fit log model on each training set, predict test
    log1 = LogisticRegression(random_state=0, max_iter=1000)
    log1.fit(cluster1_trainX, cluster1_trainy)

    log2 = LogisticRegression(random_state=0, max_iter=1000)
    log2.fit(cluster2_trainX, cluster2_trainy)
    
    log3 = LogisticRegression(random_state=0, max_iter=1000)
    log3.fit(cluster3_trainX, cluster3_trainy)
    
    log4 = LogisticRegression(random_state=0, max_iter=1000)
    log4.fit(cluster4_trainX, cluster4_trainy)
    print("local models fit")
    
    cols = ['global_log_accuracy', 'global_log_misclassified',
            'global_log_auroc', 'global_log_auprc', 'global_log_f1',
            'global_rf_accuracy', 'global_rf_misclassified', 'global_rf_auroc',
            'global_rf_auprc', 'global_rf_f1', 'global_rf_plus_accuracy',
            'global_rf_plus_misclassified', 'global_rf_plus_auroc',
            'global_rf_plus_auprc', 'global_rf_plus_f1', 'local_log_accuracy',
            'local_log_misclassified', 'local_log_auroc', 'local_log_auprc',
            'local_log_f1', 'local_rf_accuracy', 'local_rf_misclassified',
            'local_rf_auroc', 'local_rf_auprc', 'local_rf_f1',
            'local_rf_plus_accuracy', 'local_rf_plus_misclassified',
            'local_rf_plus_auroc', 'local_rf_plus_auprc', 'local_rf_plus_f1']
    
    # c1_results = pd.DataFrame(columns=cols)
    # c2_results = pd.DataFrame(columns=cols)
    # c3_results = pd.DataFrame(columns=cols)
    # c4_results = pd.DataFrame(columns=cols)
    
    local_log_preds1 = log1.predict(cluster1_testX)
    local_log_acc1 = np.mean(cluster1_testy == local_log_preds1)
    local_log_mis1 = np.sum(cluster1_testy != local_log_preds1)
    local_log_auroc1 = roc_auc_score(cluster1_testy,
                                    log1.predict_proba(cluster1_testX)[:, 1])
    local_log_auprc1 = average_precision_score(cluster1_testy,
                                    log1.predict_proba(cluster1_testX)[:, 1])
    local_log_f1score1 = f1_score(cluster1_testy, local_log_preds1)
    
    global_log_preds1 = log.predict(cluster1_testX)
    global_log_acc1 = np.mean(cluster1_testy == global_log_preds1)
    global_log_mis1 = np.sum(cluster1_testy != global_log_preds1)
    global_log_auroc1 = roc_auc_score(cluster1_testy,
                                    log.predict_proba(cluster1_testX)[:, 1])
    global_log_auprc1 = average_precision_score(cluster1_testy,
                                    log.predict_proba(cluster1_testX)[:, 1])
    global_log_f1score1 = f1_score(cluster1_testy, global_log_preds1)
    
    local_rf_preds1 = rf1.predict(cluster1_testX)
    local_rf_acc1 = np.mean(cluster1_testy == local_rf_preds1)
    local_rf_mis1 = np.sum(cluster1_testy != local_rf_preds1)
    local_rf_auroc1 = roc_auc_score(cluster1_testy,
                                    rf1.predict_proba(cluster1_testX)[:, 1])
    local_rf_auprc1 = average_precision_score(cluster1_testy,
                                    rf1.predict_proba(cluster1_testX)[:, 1])
    local_rf_f1score1 = f1_score(cluster1_testy, local_rf_preds1)
    
    global_rf_preds1 = rf.predict(cluster1_testX)
    global_rf_acc1 = np.mean(cluster1_testy == global_rf_preds1)
    global_rf_mis1 = np.sum(cluster1_testy != global_rf_preds1)
    global_rf_auroc1 = roc_auc_score(cluster1_testy,
                                    rf.predict_proba(cluster1_testX)[:, 1])
    global_rf_auprc1 = average_precision_score(cluster1_testy,
                                    rf.predict_proba(cluster1_testX)[:, 1])
    global_rf_f1score1 = f1_score(cluster1_testy, global_rf_preds1)
    
    local_rf_plus_probpreds1 = rf_plus1.predict_proba(cluster1_testX)[:, 1]
    local_rf_plus_preds1 = local_rf_plus_probpreds1 > 0.5
    local_rf_plus_acc1 = np.mean(cluster1_testy == local_rf_plus_preds1)
    local_rf_plus_mis1 = np.sum(cluster1_testy != local_rf_plus_preds1)
    local_rf_plus_auroc1 = roc_auc_score(cluster1_testy,
                                         local_rf_plus_probpreds1)
    local_rf_plus_auprc1 = average_precision_score(cluster1_testy,
                                                   local_rf_plus_probpreds1)
    local_rf_plus_f1score1 = f1_score(cluster1_testy, local_rf_plus_preds1)
    
    global_rf_plus_probpreds1 = rf_plus.predict_proba(cluster1_testX)[:, 1]
    global_rf_plus_preds1 = global_rf_plus_probpreds1 > 0.5
    global_rf_plus_acc1 = np.mean(cluster1_testy == global_rf_plus_preds1)
    global_rf_plus_mis1 = np.sum(cluster1_testy != global_rf_plus_preds1)
    global_rf_plus_auroc1 = roc_auc_score(cluster1_testy,
                                            global_rf_plus_probpreds1)
    global_rf_plus_auprc1 = average_precision_score(cluster1_testy,
                                            global_rf_plus_probpreds1)
    global_rf_plus_f1score1 = f1_score(cluster1_testy, global_rf_plus_preds1)
    
    # add row to cluster 1 results data
    c1_results = np.asarray([global_log_acc1, global_log_mis1, global_log_auroc1,
                        global_log_auprc1, global_log_f1score1, global_rf_acc1,
                        global_rf_mis1, global_rf_auroc1, global_rf_auprc1,
                        global_rf_f1score1, global_rf_plus_acc1, global_rf_plus_mis1,
                        global_rf_plus_auroc1, global_rf_plus_auprc1,
                        global_rf_plus_f1score1,                    
                        local_log_acc1, local_log_mis1,
                        local_log_auroc1, local_log_auprc1, local_log_f1score1,
                        local_rf_acc1, local_rf_mis1, local_rf_auroc1,
                        local_rf_auprc1, local_rf_f1score1, local_rf_plus_acc1,
                        local_rf_plus_mis1, local_rf_plus_auroc1,
                        local_rf_plus_auprc1, local_rf_plus_f1score1])
    print("cluster 1 results complete")
    
    local_log_preds2 = log2.predict(cluster2_testX)
    local_log_acc2 = np.mean(cluster2_testy == local_log_preds2)
    local_log_mis2 = np.sum(cluster2_testy != local_log_preds2)
    local_log_auroc2 = roc_auc_score(cluster2_testy,
                                    log2.predict_proba(cluster2_testX)[:, 1])
    local_log_auprc2 = average_precision_score(cluster2_testy,
                                    log2.predict_proba(cluster2_testX)[:, 1])
    local_log_f1score2 = f1_score(cluster2_testy, local_log_preds2)
    
    global_log_preds2 = log.predict(cluster2_testX)
    global_log_acc2 = np.mean(cluster2_testy == global_log_preds2)
    global_log_mis2 = np.sum(cluster2_testy != global_log_preds2)
    global_log_auroc2 = roc_auc_score(cluster2_testy,
                                    log.predict_proba(cluster2_testX)[:, 1])
    global_log_auprc2 = average_precision_score(cluster2_testy,
                                    log.predict_proba(cluster2_testX)[:, 1])
    global_log_f1score2 = f1_score(cluster2_testy, global_log_preds2)
    
    local_rf_preds2 = rf2.predict(cluster2_testX)
    local_rf_acc2 = np.mean(cluster2_testy == local_rf_preds2)
    local_rf_mis2 = np.sum(cluster2_testy != local_rf_preds2)
    local_rf_auroc2 = roc_auc_score(cluster2_testy,
                                    rf2.predict_proba(cluster2_testX)[:, 1])
    local_rf_auprc2 = average_precision_score(cluster2_testy,
                                    rf2.predict_proba(cluster2_testX)[:, 1])
    local_rf_f1score2 = f1_score(cluster2_testy, local_rf_preds2)
    
    global_rf_preds2 = rf.predict(cluster2_testX)
    global_rf_acc2 = np.mean(cluster2_testy == global_rf_preds2)
    global_rf_mis2 = np.sum(cluster2_testy != global_rf_preds2)
    global_rf_auroc2 = roc_auc_score(cluster2_testy,
                                    rf.predict_proba(cluster2_testX)[:, 1])
    global_rf_auprc2 = average_precision_score(cluster2_testy,
                                    rf.predict_proba(cluster2_testX)[:, 1])
    global_rf_f1score2 = f1_score(cluster2_testy, global_rf_preds2)
    
    local_rf_plus_probpreds2 = rf_plus2.predict_proba(cluster2_testX)[:, 1]
    local_rf_plus_preds2 = local_rf_plus_probpreds2 > 0.5
    local_rf_plus_acc2 = np.mean(cluster2_testy == local_rf_plus_preds2)
    local_rf_plus_mis2 = np.sum(cluster2_testy != local_rf_plus_preds2)
    local_rf_plus_auroc2 = roc_auc_score(cluster2_testy,
                                         local_rf_plus_probpreds2)
    local_rf_plus_auprc2 = average_precision_score(cluster2_testy,
                                                   local_rf_plus_probpreds2)
    local_rf_plus_f1score2 = f1_score(cluster2_testy, local_rf_plus_preds2)
    
    global_rf_plus_probpreds2 = rf_plus.predict_proba(cluster2_testX)[:, 1]
    global_rf_plus_preds2 = global_rf_plus_probpreds2 > 0.5
    global_rf_plus_acc2 = np.mean(cluster2_testy == global_rf_plus_preds2)
    global_rf_plus_mis2 = np.sum(cluster2_testy != global_rf_plus_preds2)
    global_rf_plus_auroc2 = roc_auc_score(cluster2_testy,
                                            global_rf_plus_probpreds2)
    global_rf_plus_auprc2 = average_precision_score(cluster2_testy,
                                            global_rf_plus_probpreds2)
    global_rf_plus_f1score2 = f1_score(cluster2_testy, global_rf_plus_preds2)
    
    # add row to cluster 2 results data
    c2_results = np.asarray([global_log_acc2, global_log_mis2, global_log_auroc2,
                        global_log_auprc2, global_log_f1score2, global_rf_acc2,
                        global_rf_mis2, global_rf_auroc2, global_rf_auprc2,
                        global_rf_f1score2, global_rf_plus_acc2, global_rf_plus_mis2,
                        global_rf_plus_auroc2, global_rf_plus_auprc2,
                        global_rf_plus_f1score2, 
                        local_log_acc2, local_log_mis2,
                        local_log_auroc2, local_log_auprc2, local_log_f1score2,
                        local_rf_acc2, local_rf_mis2, local_rf_auroc2,
                        local_rf_auprc2, local_rf_f1score2, local_rf_plus_acc2,
                        local_rf_plus_mis2, local_rf_plus_auroc2,
                        local_rf_plus_auprc2, local_rf_plus_f1score2])
    print("cluster 2 results complete")
    
    local_log_preds3 = log3.predict(cluster3_testX)
    local_log_acc3 = np.mean(cluster3_testy == local_log_preds3)
    local_log_mis3 = np.sum(cluster3_testy != local_log_preds3)
    local_log_auroc3 = roc_auc_score(cluster3_testy,
                                    log3.predict_proba(cluster3_testX)[:, 1])
    local_log_auprc3 = average_precision_score(cluster3_testy,
                                    log3.predict_proba(cluster3_testX)[:, 1])
    local_log_f1score3 = f1_score(cluster3_testy, local_log_preds3)
    
    global_log_preds3 = log.predict(cluster3_testX)
    global_log_acc3 = np.mean(cluster3_testy == global_log_preds3)
    global_log_mis3 = np.sum(cluster3_testy != global_log_preds3)
    global_log_auroc3 = roc_auc_score(cluster3_testy,
                                    log.predict_proba(cluster3_testX)[:, 1])
    global_log_auprc3 = average_precision_score(cluster3_testy,
                                    log.predict_proba(cluster3_testX)[:, 1])
    global_log_f1score3 = f1_score(cluster3_testy, global_log_preds3)
    
    local_rf_preds3 = rf3.predict(cluster3_testX)
    local_rf_acc3 = np.mean(cluster3_testy == local_rf_preds3)
    local_rf_mis3 = np.sum(cluster3_testy != local_rf_preds3)
    local_rf_auroc3 = roc_auc_score(cluster3_testy,
                                    rf3.predict_proba(cluster3_testX)[:, 1])
    local_rf_auprc3 = average_precision_score(cluster3_testy,
                                    rf3.predict_proba(cluster3_testX)[:, 1])
    local_rf_f1score3 = f1_score(cluster3_testy, local_rf_preds3)
    
    global_rf_preds3 = rf.predict(cluster3_testX)
    global_rf_acc3 = np.mean(cluster3_testy == global_rf_preds3)
    global_rf_mis3 = np.sum(cluster3_testy != global_rf_preds3)
    global_rf_auroc3 = roc_auc_score(cluster3_testy,
                                    rf.predict_proba(cluster3_testX)[:, 1])
    global_rf_auprc3 = average_precision_score(cluster3_testy,
                                    rf.predict_proba(cluster3_testX)[:, 1])
    global_rf_f1score3 = f1_score(cluster3_testy, global_rf_preds3)
    
    local_rf_plus_probpreds3 = rf_plus3.predict_proba(cluster3_testX)[:, 1]
    local_rf_plus_preds3 = local_rf_plus_probpreds3 > 0.5
    local_rf_plus_acc3 = np.mean(cluster3_testy == local_rf_plus_preds3)
    local_rf_plus_mis3 = np.sum(cluster3_testy != local_rf_plus_preds3)
    local_rf_plus_auroc3 = roc_auc_score(cluster3_testy,
                                         local_rf_plus_probpreds3)
    local_rf_plus_auprc3 = average_precision_score(cluster3_testy,
                                                   local_rf_plus_probpreds3)
    local_rf_plus_f1score3 = f1_score(cluster3_testy, local_rf_plus_preds3)
    
    global_rf_plus_probpreds3 = rf_plus.predict_proba(cluster3_testX)[:, 1]
    global_rf_plus_preds3 = global_rf_plus_probpreds3 > 0.5
    global_rf_plus_acc3 = np.mean(cluster3_testy == global_rf_plus_preds3)
    global_rf_plus_mis3 = np.sum(cluster3_testy != global_rf_plus_preds3)
    global_rf_plus_auroc3 = roc_auc_score(cluster3_testy,
                                            global_rf_plus_probpreds3)
    global_rf_plus_auprc3 = average_precision_score(cluster3_testy,
                                            global_rf_plus_probpreds3)
    global_rf_plus_f1score3 = f1_score(cluster3_testy, global_rf_plus_preds3)
    
    # add row to cluster 3 results data
    c3_results = np.asarray([global_log_acc3, global_log_mis3, global_log_auroc3,
                        global_log_auprc3, global_log_f1score3, global_rf_acc3,
                        global_rf_mis3, global_rf_auroc3, global_rf_auprc3,
                        global_rf_f1score3, global_rf_plus_acc3, global_rf_plus_mis3,
                        global_rf_plus_auroc3, global_rf_plus_auprc3,
                        global_rf_plus_f1score3,
                        local_log_acc3, local_log_mis3,
                        local_log_auroc3, local_log_auprc3, local_log_f1score3,
                        local_rf_acc3, local_rf_mis3, local_rf_auroc3,
                        local_rf_auprc3, local_rf_f1score3, local_rf_plus_acc3,
                        local_rf_plus_mis3, local_rf_plus_auroc3,
                        local_rf_plus_auprc3, local_rf_plus_f1score3])
    
    print("cluster 3 results complete")
    
    local_log_preds4 = log4.predict(cluster4_testX)
    local_log_acc4 = np.mean(cluster4_testy == local_log_preds4)
    local_log_mis4 = np.sum(cluster4_testy != local_log_preds4)
    local_log_auroc4 = roc_auc_score(cluster4_testy,
                                    log4.predict_proba(cluster4_testX)[:, 1])
    local_log_auprc4 = average_precision_score(cluster4_testy,
                                    log4.predict_proba(cluster4_testX)[:, 1])
    local_log_f1score4 = f1_score(cluster4_testy, local_log_preds4)
    
    global_log_preds4 = log.predict(cluster4_testX)
    global_log_acc4 = np.mean(cluster4_testy == global_log_preds4)
    global_log_mis4 = np.sum(cluster4_testy != global_log_preds4)
    global_log_auroc4 = roc_auc_score(cluster4_testy,
                                    log.predict_proba(cluster4_testX)[:, 1])
    global_log_auprc4 = average_precision_score(cluster4_testy,
                                    log.predict_proba(cluster4_testX)[:, 1])
    global_log_f1score4 = f1_score(cluster4_testy, global_log_preds4)
    
    local_rf_preds4 = rf4.predict(cluster4_testX)
    local_rf_acc4 = np.mean(cluster4_testy == local_rf_preds4)
    local_rf_mis4 = np.sum(cluster4_testy != local_rf_preds4)
    local_rf_auroc4 = roc_auc_score(cluster4_testy,
                                    rf4.predict_proba(cluster4_testX)[:, 1])
    local_rf_auprc4 = average_precision_score(cluster4_testy,
                                    rf4.predict_proba(cluster4_testX)[:, 1])
    local_rf_f1score4 = f1_score(cluster4_testy, local_rf_preds4)
    
    global_rf_preds4 = rf.predict(cluster4_testX)
    global_rf_acc4 = np.mean(cluster4_testy == global_rf_preds4)
    global_rf_mis4 = np.sum(cluster4_testy != global_rf_preds4)
    global_rf_auroc4 = roc_auc_score(cluster4_testy,
                                    rf.predict_proba(cluster4_testX)[:, 1])
    global_rf_auprc4 = average_precision_score(cluster4_testy,
                                    rf.predict_proba(cluster4_testX)[:, 1])
    global_rf_f1score4 = f1_score(cluster4_testy, global_rf_preds4)
    
    local_rf_plus_probpreds4 = rf_plus4.predict_proba(cluster4_testX)[:, 1]
    local_rf_plus_preds4 = local_rf_plus_probpreds4 > 0.5
    local_rf_plus_acc4 = np.mean(cluster4_testy == local_rf_plus_preds4)
    local_rf_plus_mis4 = np.sum(cluster4_testy != local_rf_plus_preds4)
    local_rf_plus_auroc4 = roc_auc_score(cluster4_testy,
                                         local_rf_plus_probpreds4)
    local_rf_plus_auprc4 = average_precision_score(cluster4_testy,
                                                   local_rf_plus_probpreds4)
    local_rf_plus_f1score4 = f1_score(cluster4_testy, local_rf_plus_preds4)
    
    global_rf_plus_probpreds4 = rf_plus.predict_proba(cluster4_testX)[:, 1]
    global_rf_plus_preds4 = global_rf_plus_probpreds4 > 0.5
    global_rf_plus_acc4 = np.mean(cluster4_testy == global_rf_plus_preds4)
    global_rf_plus_mis4 = np.sum(cluster4_testy != global_rf_plus_preds4)
    global_rf_plus_auroc4 = roc_auc_score(cluster4_testy,
                                            global_rf_plus_probpreds4)
    global_rf_plus_auprc4 = average_precision_score(cluster4_testy,
                                            global_rf_plus_probpreds4)
    global_rf_plus_f1score4 = f1_score(cluster4_testy, global_rf_plus_preds4)
    
    # add row to cluster 4 results data
    c4_results = np.asarray([global_log_acc4, global_log_mis4, global_log_auroc4,
                        global_log_auprc4, global_log_f1score4, global_rf_acc4,
                        global_rf_mis4, global_rf_auroc4, global_rf_auprc4,
                        global_rf_f1score4, global_rf_plus_acc4, global_rf_plus_mis4,
                        global_rf_plus_auroc4, global_rf_plus_auprc4,
                        global_rf_plus_f1score4,
                        local_log_acc4, local_log_mis4,
                        local_log_auroc4, local_log_auprc4, local_log_f1score4,
                        local_rf_acc4, local_rf_mis4, local_rf_auroc4,
                        local_rf_auprc4, local_rf_f1score4, local_rf_plus_acc4,
                        local_rf_plus_mis4, local_rf_plus_auroc4,
                        local_rf_plus_auprc4, local_rf_plus_f1score4])
    
    print("cluster 4 results complete")
    
    # combine cluster results into one array where the clusters are the rows
    result = np.vstack((c1_results, c2_results, c3_results, c4_results))
    
    # Convert the NumPy array to a pandas DataFrame
    cluster_results = pd.DataFrame(result)

    # Set the column names
    cluster_results.columns = cols

    
    # row names
    cluster_results.index = ['cluster1', 'cluster2', 'cluster3', 'cluster4']
    return cluster_results


if __name__ == '__main__':
    print("HERE")
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=0)
    log, rf, rf_plus = train_models(X_train, y_train)
    lmdi_results = lmdi_plus(X_train, y_train, X_test, y_test, log, rf, rf_plus)
    lmdi_results.to_csv('compas_output/compas_lmdi_results.csv')
    shap_results = tree_shap(X_train, y_train, X_test, y_test, log, rf, rf_plus)
    shap_results.to_csv('compas_output/compas_shap_results.csv')
    print(lmdi_results)
    print(shap_results)
    