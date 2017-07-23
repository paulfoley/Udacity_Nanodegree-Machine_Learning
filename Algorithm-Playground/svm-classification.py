'''
Playing with the Support Vector Machine Classifier
'''
## Import SVC
from sklearn.svm import SVC

## Create Training Samples and Labels
training_samples = [[0, 0], [1, 1]]
training_labels = [0, 1]

## Create Classifier
classifier = SVC(C=1.0, cache_size=200, class_weight=None, 
				coef0=0.0, decision_function_shape=None, degree=3, 
				gamma='auto', kernel='rbf', max_iter=-1,
				probability=False, random_state=None, shrinking=True,
    			tol=0.001, verbose=False)

## Fit Classifier
classifier.fit(training_samples, training_labels)  

## Predict
prediction = classifier.predict([[2., 2.]])

## Get Support Vectors
support_vectors = classifier.support_vectors_

## Get Indices of Support Vectors
support = classifier.support_ 

## Get Number of Support Vectors for Each Class
n_support = classifier.n_support_ 

## Output
print("Prediction:")
print(prediction)
print("\nSupport Vectors:")
print(support_vectors)
print("\nSupport Vector Indices:")
print(support)
print("\nNumber of Support Vectors:")
print(n_support)
