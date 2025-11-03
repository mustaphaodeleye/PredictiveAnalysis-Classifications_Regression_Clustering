# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load your dataset (replace 'synthetic_healthcare_data.csv' with your actual dataset)
data = pd.read_csv('synthetic_healthcare_data.csv')

# Assuming 'readmission' is the target column and the rest are features
X = data.drop('readmission', axis=1)  # Features
y = data['readmission']  # Target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize the data (important for Ridge regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Ridge regression model
ridge_regressor = Ridge(alpha=1.0)  # alpha is the regularization parameter

# Train the model
ridge_regressor.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = ridge_regressor.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# 1. Scatter plot of predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values (Readmission)')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()

# 2. Histogram of residuals (errors)
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='red', bins=20)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 3. Plot the coefficients (importance of features)
coefficients = ridge_regressor.coef_
features = X.columns
plt.figure(figsize=(8, 6))
plt.barh(features, coefficients, color='green')
plt.title('Feature Coefficients (Ridge Regression Analysis)')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.grid(True)
plt.show()