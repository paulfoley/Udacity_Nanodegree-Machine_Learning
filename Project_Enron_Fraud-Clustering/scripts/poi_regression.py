"""
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data
"""    

# Python Imports
import pickle
import sys
import matplotlib.pyplot as plt

# Import sklearn Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Local Imports
from feature_format import featureFormat, targetFeatureSplit

# Create Dictionary
dictionary = pickle.load(open("dataset_modified.pkl", "r") )

# List the features
# First item in the list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

# training-testing split needed in regression
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

# Create Regression
reg = LinearRegression(fit_intercept=True)
reg.fit(feature_train, target_train)

# Output
print("Regression Coefficient and Intercept:")
print(reg.coef_, reg.intercept_)
score = reg.score(feature_test, target_test)
print('\nRegression Score:')
print(score)

# draw the scatterplot, with color-coded training and testing points
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

# labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")

# Draw the regression line
plt.plot( feature_test, reg.predict(feature_test) )

# Use Testing Data
reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="b") 
print('\nRegression Coefficient on Testing Data:')
print(reg.coef_)
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
