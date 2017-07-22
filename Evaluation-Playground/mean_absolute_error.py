## Example of Mean Absolute Error

# Imports
import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae

# Load the dataset
linnerud_data = load_linnerud()

# Create Features
features = linnerud_data.data

# Create Labels
labels = linnerud_data.target

# Split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
features_train, features_test, labels_train, labels_test = train_test_split(features, labels)

##### Decision Tree Regression #####
# Create Regression
reg_dt = DecisionTreeRegressor()

# Fit Regression
reg_dt.fit(features_train, labels_train)

# Predict on Regression
predict_dt = reg_dt.predict(features_test)

# Mean Absolute Error
mean_absolute_error_dt = mae(labels_test, predict_dt)

##### Linear Regression #####
# Create Regression
reg_l = LinearRegression()

# Fit Regression
reg_l.fit(features_train, labels_train)

# Predict on Regression
predict_l = reg_l.predict(features_test)

# Mean Absolute Error
mean_absolute_error_l = mae(labels_test, predict_l)

# Results
results = {
 "Linear Regression": mean_absolute_error_l,
 "Decision Tree": mean_absolute_error_dt
}

print(results)