""" 
    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
# Imports
from time import time
import sys
sys.path.append("/Users/Nexu/Desktop/Machine_Learning-Projects/Project_Enron/tools/")
from email_preprocess import preprocess

'''
features_train and features_test are the features for the training and testing datasets, respectively
labels_train and labels_test are the corresponding item labels
'''
features_train, features_test, labels_train, labels_test = preprocess()

# Decrease the Training Size for Faster Processing
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100] 

########################## SVM #################################
#Import Statement and SVC creation
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create Classifier
classifier = SVC(kernel="rbf", C = 10000.0)

# Start timer
t0 = time()

# Fit the Classifier on the Training Features and Labels
classifier.fit(features_train, labels_train)
print("training time: " + str(round(time()-t0, 3)) + 's')

# Use the Trained Classifier to Predict Labels for the Test Features
predictions = classifier.predict(features_test)
print('prediction time: ' + str(round(time()-t0, 3)) + 's')

# Calculate and Return the Accuracy on the Test Data
accuracy = accuracy_score(predictions, labels_test)
print("accuracy time: " + str(round(time()-t0, 3)) + 's')

# Output Results
count = 0
for prediction in predictions:
	if prediction == 1:
		count += 1

print(count)

