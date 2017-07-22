"""
Create the decision boundary and make a plot 
that visually shows the decision boundary
"""

# Imports
from sklearn.naive_bayes import GaussianNB
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
import numpy as np
import pylab as pl

# Create the Training Features and Labels, as well as the Testing Features and Labels
features_train, labels_train, features_test, labels_test = makeTerrainData()

'''
The training data (features_train, labels_train)
have both "fast" and "slow" points mixed in together

Separate them so we can give them different colors in the scatterplot,
and visually identify them
'''
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

# Create Classifier
classifier = GaussianNB()

# Fit the Classifier on the training features and labels
classifier = classifier.fit(features_train, labels_train)

    # Use the trained classifier to predict labels for the test features
    prediction = classifier.predict(features_test)

    # Calculate and return the accuracy on the test data
    accuracy = classifier.score(features_test, labels_test)

# Draw the decision boundary with the text points overlaid
prettyPicture(classifier, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())