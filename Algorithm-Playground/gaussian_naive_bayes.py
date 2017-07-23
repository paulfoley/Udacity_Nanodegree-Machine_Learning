"""
Playing with Gaussian Naive Bayes (GaussianNB)
"""
## Import Numpy
import numpy as np

## Import Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

## Create Features & Labels
features = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
labels = np.array([1, 1, 1, 2, 2, 2])

## Create Classifier
classifier = GaussianNB()

## Fit Classifier
classifier.fit(features, labels)

## Predict
prediction = classifier.predict([[-0.8, -1]])

## Create Partial Fit Classifier and Fit
classifiers_partial_fit = GaussianNB()
classifiers_partial_fit.partial_fit(features, labels, np.unique(labels))

## Predict
prediciton_partial_fit = classifiers_partial_fit.predict([[-0.8, -1]])

## output
print(prediction)
print(prediciton_partial_fit)