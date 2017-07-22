""" 
    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
# Imports 
from time import time
from sklearn.naive_bayes import GaussianNB
import sys
sys.path.append("/Users/Nexu/Desktop/Machine_Learning-Projects/Project_Enron/tools")
from email_preprocess import preprocess

'''
features_train and features_test are the features for the training and testing datasets, 
respectively, labels_train and labels_test are the corresponding item labels
'''
features_train, features_test, labels_train, labels_test = preprocess()

# Create Classifier
classifier = GaussianNB()

# Start timer
t0 = time()

# Fit the Classifier on the Training Features and Labels
classifier.fit(features_train, labels_train)
print("training time: " + str(round(time()-t0, 3)) + 's')

# Use the Trained Classifier to Predict Labels for the Test Features
prediction = classifier.predict(features_test)
print("prediction time: " + str(round(time()-t0, 3)) + 's')

# Calculate and Return the Accuracy on the Test Data
accuracy = classifier.score(features_test, labels_test)
print("accuracy time: " + str(round(time()-t0, 3)) + 's')

print(accuracy)
