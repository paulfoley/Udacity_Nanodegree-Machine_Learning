import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData
from classifyDT import classify

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

from sklearn import tree

features_train, labels_train, features_test, labels_test = makeTerrainData()

# The classify() function in classifyDT is where the magic happens
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features_train, labels_train)

# Get Accuracy Score
score = classifier.score(features_test, labels_test)


# Output Results
prettyPicture(classifier, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
print(score)