## Eample of using Regression

# Imports
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData
from sklearn import linear_model

# Create Training Featues and Labels, as well as Testing Features and Labels
features_train, labels_train, features_test, labels_test = makeTerrainData()

# Create Regression
regression = linear_model.LinearRegression()

# Fit Regression
regression.fit(features_train, labels_train)

# Plot Output
prettyPicture(regression, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())