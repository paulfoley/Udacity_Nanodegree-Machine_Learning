# In this and the following exercises, you'll be adding train test splits to the data
# to see how it changes the performance of each classifier
#
# The code provided will load the Titanic dataset like you did in project 0, then train
# a decision tree (the method you used in your project) and a Bayesian classifier (as
# discussed in the introduction videos). You don't need to worry about how these work for
# now. 
#
# What you do need to do is import a train/test split, train the classifiers on the
# training data, and store the resulting accuracy scores in the dictionary provided.

# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('titanic_data.csv')

# Limit to numeric data
data = data._get_numeric_data()

# Separate the labels
labels = data['Survived']

# Remove labels from the inputs, and age due to missing data
del data['Age'], data['Survived']

# Create Features
features = data

# Split the data into training and testing sets, 
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
features_train, features_test, labels_train, labels_test = train_test_split(features, labels)

##### Decision Tree #####
# Create Classifier
clf_dt = DecisionTreeClassifier()

# Fit Classifier
clf_dt.fit(features_train, labels_train)

# Predict on Classifier
predict_dt = clf_dt.predict(features_test)

# Get Accuracy Score
score_dt = accuracy_score(labels_test, predict_dt)

##### Naive Bayes #####
# Create Classifier
clf_nb = GaussianNB()

# Fit Classifier
clf_nb.fit(features_train, labels_train)

# Predict on Classifier
predict_nb = clf_nb.predict(features_test)

# Get Accuracy Score
score_nb = accuracy_score(labels_test, predict_nb)

# Answer
answer = { 
			"Naive Bayes Score": score_nb, 
			"Decision Tree Score": score_dt
}

print(answer)