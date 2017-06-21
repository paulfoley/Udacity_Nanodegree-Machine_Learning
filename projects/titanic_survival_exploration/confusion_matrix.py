# Train two classifiers and look at their confusion matrices.
# Your job is to create a train/test split in the data
# and report the results in the dictionary at the bottom.

# Imports
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

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
# using the default settings for train_test_split (or test_size = 0.25 if specified).
# Then, train and test the classifiers with your newly split data instead of X and y.
features_train, features_test, labels_train, labels_test = train_test_split(features, labels)

# Decision Tree Classifier
clf_dt = DecisionTreeClassifier()
clf_dt.fit(features_train, labels_train)
predict_dt = clf_dt.predict(features_test)
confusion_matrix_dt = confusion_matrix(labels_test, predict_dt)

# Naive Bayes Classifier
clf_nb = GaussianNB()
clf_nb.fit(features_train, labels_train)
predict_nb = clf_nb.predict(features_test)
confusion_matrix_nb = confusion_matrix(labels_test, predict_nb)

# Answer
confusions = {
				"Naive Bayes": confusion_matrix_nb,
				"Decision Tree": confusion_matrix_dt
}

print(confusions)