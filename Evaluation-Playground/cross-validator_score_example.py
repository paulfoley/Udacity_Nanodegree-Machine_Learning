# Imports
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC

# Import Iris Data
iris = datasets.load_iris()

# Create Training Features and Labels, as well as Testing Features and Labels
feature_train, feature_test, label_train, label_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

# Create and Fit Classifier
classifier = SVC(kernel = 'linear', C = 1).fit(feature_train, label_train)

# Get Accuracy Score
score = classifier.score(feature_test, label_test)
print(score)

# Comput Cross-Validated Score
scores = cross_val_score(classifier, iris.data, iris.target, cv = 5)
print(scores)