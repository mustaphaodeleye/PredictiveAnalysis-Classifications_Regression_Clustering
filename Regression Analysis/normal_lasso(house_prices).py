import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

#Load dataset
house_data = pd.read_csv('house_prices.csv')

#One-hot encode the 'location'column
house_data = pd.get_dummies(house_data, columns=['location'], drop_first=True)

#Define features and target
X = house_data[['size', 'bedrooms', 'location_B', 'location_C']]
y = house_data['price']

#Split data imto training anf testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize and train the Lasso model
scaler = StandardScaler()
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

#Evaluate the model
scaler = StandardScaler()
model = Lasso(alpha=0.1) #You can adjust the alpha value (regularization strength)
model.fit(X_train, y_train)

#Evaluate the model
score = model.score(X_test, y_test)
print(f'R-squared: {score}')

#Predictions
y_pred = model.predict(X_test)

#Ploting  the actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--') #Ideal line
plt.title('Actual vs Predicted House price')
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.show()


