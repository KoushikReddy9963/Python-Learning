from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

housing_data = fetch_california_housing()

df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)

# Add the target column to the DataFrame
df['target'] = housing_data.target

# Divide the dataset into independent (X) and dependent (Y) variables
X = df.iloc[:, :-1]  # independent features
Y = df.iloc[:, -1]  # dependent feature

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
mse = cross_val_score(lin_reg, X, Y, scoring='neg_mean_squared_error', cv=5)
mean_mse = np.mean(mse)
print(f"Linear Regression MSE: {mean_mse}")

# Ridge Regression
ridge = Ridge()
params = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
ridge_regression = GridSearchCV(ridge, params, scoring='neg_mean_squared_error', cv=5)
ridge_regression.fit(X, Y)
print(f"Ridge Regression Best Params: {ridge_regression.best_params_}")
print(f"Ridge Regression Best Score: {ridge_regression.best_score_}")

# Lasso Regression
lasso = Lasso()
params_lasso = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
lasso_regression = GridSearchCV(lasso, params_lasso, scoring='neg_mean_squared_error', cv=5)
lasso_regression.fit(X, Y)
print(f"Lasso Regression Best Params: {lasso_regression.best_params_}")
print(f"Lasso Regression Best Score: {lasso_regression.best_score_}")

# Logistic Regression using L1 and L2 norms
# For Logistic Regression, we need a classification target, so we create a binary target.
# We will use a simple condition to convert the target to binary (e.g., median split).
Y_binary = (df['target'] > df['target'].median()).astype(int)

# Split data into train and test for Logistic Regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_binary, test_size=0.3, random_state=42)

log_reg_l1 = LogisticRegression(penalty='l1', solver='liblinear')
log_reg_l1.fit(X_train, Y_train)
Y_pred_l1 = log_reg_l1.predict(X_test)
accuracy_l1 = accuracy_score(Y_test, Y_pred_l1)
print(f"Logistic Regression (L1) Accuracy: {accuracy_l1}")

log_reg_l2 = LogisticRegression(penalty='l2', solver='liblinear')
log_reg_l2.fit(X_train, Y_train)
Y_pred_l2 = log_reg_l2.predict(X_test)
accuracy_l2 = accuracy_score(Y_test, Y_pred_l2)
print(f"Logistic Regression (L2) Accuracy: {accuracy_l2}")
