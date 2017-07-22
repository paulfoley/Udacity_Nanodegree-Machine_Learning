# Example of Support Vector Machine Classifier

# Import SVM from sklean module
from sklearn.svm import SVC

# Training Samples and Labels
training_samples = [[0, 0], [1, 1]]
training_labels = [0, 1]

# Create Classifier
classifier = SVC(C=1.0, cache_size=200, class_weight=None, 
				coef0=0.0, cdecision_function_shape=None, degree=3, 
				gamma='auto', kernel='rbf', max_iter=-1,
				probability=False, random_state=None, shrinking=True,
    			tol=0.001, verbose=False)

# Fit Samples to Labels
classifier.fit(training_samples, training_labels)  

# Predict
prediction = classifier.predict([[2., 2.]])
print(prediction)

# Get Support Vectors
support_vectors = classifier.support_vectors_
print(support_vectors)

# Get Indices of Support Vectors
support = classifier.support_ 
print(support)

# Get Number of Support Vectors for Each Class
n_support = classifier.n_support_ 
print(n_support)

