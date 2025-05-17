from imodels.tree.rf_plus.rf_plus.rf_plus_models import RandomForestPlusRegressor
from sklearn.ensemble import RandomForestRegressor
from imodels.tree.rf_plus.feature_importance.rfplus_explainer import *
import time

def main():
    X = np.random.rand(500000, 1000)
    y = np.random.rand(500000)
    start = time.time()
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.33, random_state=42)
    rf.fit(X, y)
    end = time.time()
    print("Time to fit random forest: ", end - start)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    start = time.time()
    rf_plus = RandomForestPlusRegressor(rf)
    rf_plus.fit(X_train, y_train)
    end = time.time()
    print("Time to fit random forest plus: ", end - start)
    start = time.time()
    lmdi_explainer = AloRFPlusMDI(rf_plus, evaluate_on = 'oob')
    train_feature_importance =lmdi_explainer.explain_subtract_intercept(X_train, y_train)
    test_feature_importance = lmdi_explainer.explain_subtract_intercept(X=X_test, y=None)
    print("Train feature importance: ", train_feature_importance[:20])
    print("Test feature importance: ", test_feature_importance[:20])
    end = time.time()
    print("Time to compute feature importance: ", end - start)


if __name__ == "__main__":
    main()