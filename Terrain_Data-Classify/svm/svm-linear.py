import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

########################## SVM #################################
#Import Statement and SVC creation
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create Classifier
classifier = SVC(kernel="linear")

# Fit the Classifier on the Training Features and Labels
classifier.fit(features_train, labels_train)

# Use the Trained Classifier to Predict Labels for the Test Features
prediction = classifier.predict(features_test)

# Calculate and Return the Accuracy on the Test Data
accuracy = accuracy_score(prediction, labels_test)

# Output Results
prettyPicture(classifier, features_test, labels_test)
print(accuracy)