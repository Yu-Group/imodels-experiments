rf_plus_ridge = RandomForestPlusClassifier(rf_model=est, prediction_model=LogisticRegressionCV(penalty='l2', cv=5, max_iter=10000, random_state=0))
rf_plus_ridge.fit(X_train, y_train)

rf_plus_lasso = RandomForestPlusClassifier(rf_model=est, prediction_model=LogisticRegressionCV(penalty='l1', solver = 'saga', cv=3, n_jobs=-1, tol=5e-4, max_iter=5000, random_state=0))
rf_plus_lasso.fit(X_train, y_train)

rf_plus_ridge = RandomForestPlusRegressor(rf_model=est, prediction_model=RidgeCV(cv=5))
rf_plus_ridge.fit(X_train, y_train)

rf_plus_lasso = RandomForestPlusRegressor(rf_model=est, prediction_model=LassoCV(cv=5, max_iter=10000, random_state=0))
rf_plus_lasso.fit(X_train, y_train)