import imodels
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, mean_squared_error, r2_score
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import *
from sklearn.preprocessing import StandardScaler
import copy
import matplotlib.pyplot as plt
import openml
from sklearn.linear_model import Ridge



def ablation_removal_negative(train_mean, data, feature_importance, feature_importance_rank, feature_index):
    data_copy = data.copy()
    indices = feature_importance_rank[:, feature_index]
    sum = 0
    for i in range(data.shape[0]):
        if feature_importance[i, indices[i]] < 0:
            sum += 1
            data_copy[i, indices[i]] = train_mean[indices[i]]
    print("Remove sum: ", sum)
    return data_copy


def ablation_removal_positive(train_mean, data, feature_importance, feature_importance_rank, feature_index):
    data_copy = data.copy()
    indices = feature_importance_rank[:, feature_index]
    sum = 0
    for i in range(data.shape[0]):
        if feature_importance[i, indices[i]] > 0:
            sum += 1
            data_copy[i, indices[i]] = train_mean[indices[i]]
    print("Remove sum: ", sum)
    return data_copy

def main():
    X, y, _ = imodels.get_clean_dataset("diabetes_regr")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    est = RandomForestRegressor(n_estimators=1, min_samples_leaf=5, bootstrap=True, max_features=0.33, random_state=42)
    est.fit(X_train, y_train)

    rf_plus_base_inbag = RandomForestPlusRegressor(rf_model=est, include_raw=False, fit_on="inbag", prediction_model=LinearRegression())
    rf_plus_base_inbag.fit(X_train, y_train)

    rf_plus_base_inbag_with_raw = RandomForestPlusRegressor(rf_model=est, include_raw=True, fit_on="inbag", prediction_model=LinearRegression())
    rf_plus_base_inbag_with_raw.fit(X_train, y_train)

    shap_explainer = shap.TreeExplainer(est)
    rf_plus_explainer = RFPlusMDI(rf_plus_base_inbag, evaluate_on="inbag", mode="only_k")
    rf_plus_with_raw_explainer = RFPlusMDI(rf_plus_base_inbag_with_raw, evaluate_on="inbag", mode="only_k")

    shap_train = shap_explainer.shap_values(X_train, check_additivity=False)
    shap_train_neg_rank = np.argsort(shap_train)
    shap_train_pos_rank = np.argsort(-shap_train)
    shap_test = shap_explainer.shap_values(X_test, check_additivity=False)
    shap_test_neg_rank = np.argsort(shap_test)
    shap_test_pos_rank = np.argsort(-shap_test)

    rf_plus_train = rf_plus_explainer.explain_linear_partial(X=X_train, y=None)
    rf_plus_train_neg_rank = np.argsort(rf_plus_train)
    rf_plus_train_pos_rank = np.argsort(-rf_plus_train)
    rf_plus_test = rf_plus_explainer.explain_linear_partial(X=X_test, y=None)
    rf_plus_test_neg_rank = np.argsort(rf_plus_test)
    rf_plus_test_pos_rank = np.argsort(-rf_plus_test)

    rf_plus_train_with_raw = rf_plus_with_raw_explainer.explain_linear_partial(X=X_train, y=None)
    rf_plus_train_with_raw_neg_rank = np.argsort(rf_plus_train_with_raw)
    rf_plus_train_with_raw_pos_rank = np.argsort(-rf_plus_train_with_raw)
    rf_plus_test_with_raw = rf_plus_with_raw_explainer.explain_linear_partial(X=X_test, y=None)
    rf_plus_test_with_raw_neg_rank = np.argsort(rf_plus_test_with_raw)
    rf_plus_test_with_raw_pos_rank = np.argsort(-rf_plus_test_with_raw)

    y_pred_mean_shap_train_pos = []
    y_pred_mean_shap_train_neg = []
    y_pred_mean_rf_plus_train_pos = []
    y_pred_mean_rf_plus_train_neg = []
    y_pred_mean_rf_plus_with_raw_train_pos = []
    y_pred_mean_rf_plus_with_raw_train_neg = []

    y_pred_mean_shap_test_pos = []
    y_pred_mean_shap_test_neg = []
    y_pred_mean_rf_plus_test_pos = []
    y_pred_mean_rf_plus_test_neg = []
    y_pred_mean_rf_plus_with_raw_test_pos = []
    y_pred_mean_rf_plus_with_raw_test_neg = []

    #train_mean = np.mean(X_train, axis=0)
    train_mean = np.zeros(X_train.shape[1])
    ablation_model = est

    # Shap Train
    ablation_data = copy.deepcopy(X_train)
    y_pred_before = ablation_model.predict(ablation_data)
    y_pred_mean_shap_train_pos.append(np.mean(y_pred_before))
    for i in range(ablation_data.shape[1]):
        ablation_data = ablation_removal_positive(train_mean, ablation_data, shap_train, shap_train_pos_rank, i)
        y_pred = ablation_model.predict(ablation_data)
        y_pred_mean_shap_train_pos.append(np.mean(y_pred))

    ablation_data = copy.deepcopy(X_train)
    y_pred_before = ablation_model.predict(ablation_data)
    y_pred_mean_shap_train_neg.append(np.mean(y_pred_before))
    for i in range(ablation_data.shape[1]):
        ablation_data = ablation_removal_negative(train_mean, ablation_data, shap_train, shap_train_neg_rank, i)
        y_pred = ablation_model.predict(ablation_data)
        y_pred_mean_shap_train_neg.append(np.mean(y_pred))
    
    # Shap Test
    ablation_data = copy.deepcopy(X_test)
    y_pred_before = ablation_model.predict(ablation_data)
    y_pred_mean_shap_test_pos.append(np.mean(y_pred_before))
    for i in range(ablation_data.shape[1]):
        ablation_data = ablation_removal_positive(train_mean, ablation_data, shap_test, shap_test_pos_rank, i)
        y_pred = ablation_model.predict(ablation_data)
        y_pred_mean_shap_test_pos.append(np.mean(y_pred))
    
    ablation_data = copy.deepcopy(X_test)
    y_pred_before = ablation_model.predict(ablation_data)
    y_pred_mean_shap_test_neg.append(np.mean(y_pred_before))
    for i in range(ablation_data.shape[1]):
        ablation_data = ablation_removal_negative(train_mean, ablation_data, shap_test, shap_test_neg_rank, i)
        y_pred = ablation_model.predict(ablation_data)
        y_pred_mean_shap_test_neg.append(np.mean(y_pred))

    # RF Plus Train
    ablation_data = copy.deepcopy(X_train)
    y_pred_before = ablation_model.predict(ablation_data)
    y_pred_mean_rf_plus_train_pos.append(np.mean(y_pred_before))
    for i in range(ablation_data.shape[1]):
        ablation_data = ablation_removal_positive(train_mean, ablation_data, rf_plus_train, rf_plus_train_pos_rank, i)
        y_pred = ablation_model.predict(ablation_data)
        y_pred_mean_rf_plus_train_pos.append(np.mean(y_pred))

    ablation_data = copy.deepcopy(X_train)
    y_pred_before = ablation_model.predict(ablation_data)
    y_pred_mean_rf_plus_train_neg.append(np.mean(y_pred_before))
    for i in range(ablation_data.shape[1]):
        ablation_data = ablation_removal_negative(train_mean, ablation_data, rf_plus_train, rf_plus_train_neg_rank, i)
        y_pred = ablation_model.predict(ablation_data)
        y_pred_mean_rf_plus_train_neg.append(np.mean(y_pred))

    # RF Plus Test
    ablation_data = copy.deepcopy(X_test)
    y_pred_before = ablation_model.predict(ablation_data)
    y_pred_mean_rf_plus_test_pos.append(np.mean(y_pred_before))
    for i in range(ablation_data.shape[1]):
        ablation_data = ablation_removal_positive(train_mean, ablation_data, rf_plus_test, rf_plus_test_pos_rank, i)
        y_pred = ablation_model.predict(ablation_data)
        y_pred_mean_rf_plus_test_pos.append(np.mean(y_pred))

    ablation_data = copy.deepcopy(X_test)
    y_pred_before = ablation_model.predict(ablation_data)
    y_pred_mean_rf_plus_test_neg.append(np.mean(y_pred_before))
    for i in range(ablation_data.shape[1]):
        ablation_data = ablation_removal_negative(train_mean, ablation_data, rf_plus_test, rf_plus_test_neg_rank, i)
        y_pred = ablation_model.predict(ablation_data)
        y_pred_mean_rf_plus_test_neg.append(np.mean(y_pred))

    # RF Plus with Raw Train
    ablation_data = copy.deepcopy(X_train)
    y_pred_before = ablation_model.predict(ablation_data)
    y_pred_mean_rf_plus_with_raw_train_pos.append(np.mean(y_pred_before))
    for i in range(ablation_data.shape[1]):
        ablation_data = ablation_removal_positive(train_mean, ablation_data, rf_plus_train_with_raw, rf_plus_train_with_raw_pos_rank, i)
        y_pred = ablation_model.predict(ablation_data)
        y_pred_mean_rf_plus_with_raw_train_pos.append(np.mean(y_pred))

    ablation_data = copy.deepcopy(X_train)
    y_pred_before = ablation_model.predict(ablation_data)
    y_pred_mean_rf_plus_with_raw_train_neg.append(np.mean(y_pred_before))
    for i in range(ablation_data.shape[1]):
        ablation_data = ablation_removal_negative(train_mean, ablation_data, rf_plus_train_with_raw, rf_plus_train_with_raw_neg_rank, i)
        y_pred = ablation_model.predict(ablation_data)
        y_pred_mean_rf_plus_with_raw_train_neg.append(np.mean(y_pred))

    # RF Plus with Raw Test
    ablation_data = copy.deepcopy(X_test)
    y_pred_before = ablation_model.predict(ablation_data)
    y_pred_mean_rf_plus_with_raw_test_pos.append(np.mean(y_pred_before))
    for i in range(ablation_data.shape[1]):
        ablation_data = ablation_removal_positive(train_mean, ablation_data, rf_plus_test_with_raw, rf_plus_test_with_raw_pos_rank, i)
        y_pred = ablation_model.predict(ablation_data)
        y_pred_mean_rf_plus_with_raw_test_pos.append(np.mean(y_pred))

    ablation_data = copy.deepcopy(X_test)
    y_pred_before = ablation_model.predict(ablation_data)
    y_pred_mean_rf_plus_with_raw_test_neg.append(np.mean(y_pred_before))
    for i in range(ablation_data.shape[1]):
        ablation_data = ablation_removal_negative(train_mean, ablation_data, rf_plus_test_with_raw, rf_plus_test_with_raw_neg_rank, i)
        y_pred = ablation_model.predict(ablation_data)
        y_pred_mean_rf_plus_with_raw_test_neg.append(np.mean(y_pred))

    # make all positive in one plot
    plt.plot(y_pred_mean_shap_train_pos, label="Shap Train")
    plt.plot(y_pred_mean_rf_plus_train_pos, label="RF Plus Train")
    plt.plot(y_pred_mean_rf_plus_with_raw_train_pos, label="RF Plus with Raw Train")
    plt.legend()
    plt.title("All Positive Train")
    plt.savefig("all_positive_train.png")
    plt.clf()

    plt.plot(y_pred_mean_shap_test_pos, label="Shap Test")
    plt.plot(y_pred_mean_rf_plus_test_pos, label="RF Plus Test")
    plt.plot(y_pred_mean_rf_plus_with_raw_test_pos, label="RF Plus with Raw Test")
    plt.legend()
    plt.title("All Positive Test")
    plt.savefig("all_positive_test.png")
    plt.clf()

    # make all negative in one plot
    plt.plot(y_pred_mean_shap_train_neg, label="Shap Train")
    plt.plot(y_pred_mean_rf_plus_train_neg, label="RF Plus Train")
    plt.plot(y_pred_mean_rf_plus_with_raw_train_neg, label="RF Plus with Raw Train")
    plt.legend()
    plt.title("All Negative Train")
    plt.savefig("all_negative_train.png")
    plt.clf()

    plt.plot(y_pred_mean_shap_test_neg, label="Shap Test")
    plt.plot(y_pred_mean_rf_plus_test_neg, label="RF Plus Test")
    plt.plot(y_pred_mean_rf_plus_with_raw_test_neg, label="RF Plus with Raw Test")
    plt.legend()
    plt.title("All Negative Test")
    plt.savefig("all_negative_test.png")
    plt.clf()

if __name__ == "__main__":
    main()
