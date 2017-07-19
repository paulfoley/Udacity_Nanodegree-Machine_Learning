# Look at the performance of a couple of classifiers using the Titanic dataset
# Add a train/test split, then store the results in the dictionary provided.

# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision

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
# Using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
features_train, features_test, labels_train, labels_test = train_test_split(features, labels)


##### Decision Tree #####
# Create Classifier
clf_dt = DecisionTreeClassifier()

# Fit Classifier
clf_dt.fit(features_train, labels_train)

# Predict on Classifier
predict_dt = clf_dt.predict(features_test)

# Get Recall & Precision Scores
recall_dt = recall(labels_test, predict_dt)
precision_dt = precision(labels_test, predict_dt)


##### Naive Bayes #####
# Create Classifier
clf_nb = GaussianNB()

# Fit Classifier
clf_nb.fit(features_train, labels_train)

# Predict on Classifier
predict_nb = clf_nb.predict(features_test)

# Get Recall & Precision Scores
recall_nb = recall(labels_test, predict_nb)
precision_nb = precision(labels_test, predict_nb)

results = {
  "Naive Bayes Recall": recall_nb,
  "Naive Bayes Precision": precision_nb,
  "Decision Tree Recall": recall_dt,
  "Decision Tree Precision": precision_dt
}

print(results)