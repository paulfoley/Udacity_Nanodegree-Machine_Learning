"""
Gaussian Naive Bayes (GaussianNB)
Can perform online updates to model parameters via partial_fit method.

For details on algorithm used to update feature means and variance online, 
see Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:
"""
# Import Numpy and Gaussian NB
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Create Features
features = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
labels = np.array([1, 1, 1, 2, 2, 2])

# Create Classifier
classifiers = GaussianNB()

# Fit Classifier
classifiers.fit(features, labels)

# Predict
predictions = classifiers.predict([[-0.8, -1]])
print(predictions)

# Create Partial Fit Classifier and Fit
classifiers_partial_fit = GaussianNB()
classifiers_partial_fit.partial_fit(features, labels, np.unique(labels))

# Predict
predicitons_partial_fit = classifiers_partial_fit.predict([[-0.8, -1]])
print(predicitons_partial_fit )