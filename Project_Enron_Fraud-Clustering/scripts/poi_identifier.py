# Use a Decision Tree Classifier to identify POI's

# Python Imports
import pickle
import sys

# sklearn Imports
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

# Local Imports
from feature_format import featureFormat, targetFeatureSplit

# Load the dictionary containing the dataset
with open("dataset.pkl", "r") as data_file:
	data_dict = pickle.load(data_file)

# Add more features to features_list!
features_list = ["poi", "salary"]

# Get Training Features and Labels, as well as Testing Featrues and Labels
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

########## Decision Tree Classifier ##########
# Create Classifier
classifier = tree.DecisionTreeClassifier()

# Fit Classifier
classifier = classifier.fit(features_train, labels_train)

# Create Predcitions
predictions = classifier.predict(features_test, labels_test)

# Create Accuracy
score = classifier.score(features_test, labels_test)

# Recall & Precision
recall = recall_score(labels_test, predictions)
precision = precision_score(labels_test, predictions)

# Output
print("Number of Predicted POI's:")
print(sum(predictions))
print("\nTotal Number of People in Data Set")
print(len(predictions))
print("\nRecall Score")
print(recall)
print("\nPrecision Score")
print(precision)
