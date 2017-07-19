# Get a reliable F1 score from two classifiers, 
# and save it the scores in dictionaries.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

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

# Get F1 Score
f1_score_dt = f1_score(labels_test, predict_dt)

##### Naive Bayes #####
# Create Classifier
clf_nb = GaussianNB()

# Fit Classifier
clf_nb.fit(features_train, labels_train)

# Predict on Classifier
predict_nb = clf_nb.predict(features_test)

# Get F1 Score
f1_score_nb = f1_score(labels_test, predict_nb)

F1_scores = {
 "Naive Bayes": f1_score_nb,
 "Decision Tree": f1_score_dt
}

print(F1_scores)
