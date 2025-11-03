#Import necessary libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error, r2_score


#Load the California dataset
california = fetch_california_housing()
X = california.data
Y = california.target

#Split data into training and testing
X_train, X_test, Y_train, Y_test = (
    train_test_split(X, Y, test_size= 0.2, random_state=42))

#Initialize and train the model
model = LinearRegression()
model.fit(X_train, Y_train)

#Make predictions on the test set
predictions = model.predict(X_test)

#Calculate metrics
mse = mean_squared_error(Y_test, predictions)
r2 = r2_score(Y_test, predictions)

#Display results
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Predictions:", predictions [:5]) #Show first 5 predictions