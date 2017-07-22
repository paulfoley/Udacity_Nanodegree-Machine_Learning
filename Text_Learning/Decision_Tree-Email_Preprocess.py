""" 
    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
from time import time
from sklearn import tree
import sys
sys.path.append("../tools/")
from email_preprocess import preprocess

'''
features_train and features_test are the features for the training and testing datasets, respectively
labels_train and labels_test are the corresponding item labels
'''
features_train, features_test, labels_train, labels_test = preprocess()

# Create Classifier
classifier = tree.DecisionTreeClassifier(min_samples_split = 40)

# Fit Classifier
classifier = classifier.fit(features_train, labels_train)

# Predict
predictions = classifier.predict(features_test)

# Get Accuracy Score
score = classifier.score(features_test, labels_test)

# Number of Features
number = len(features_train[0])
print(score)