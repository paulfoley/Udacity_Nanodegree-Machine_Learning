# Use GridSearch CV to Optimize Parameters

# Imports
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import datasets

# Load Iris Data Set
iris = datasets.load_iris()

# Create Initial Parameters
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

# Create Classifier
svr = SVC()

# Run GridSearchCV to Optimizer Parameters
classifier = GridSearchCV(svr, parameters)
classifier.fit(iris.data, iris.target)
parameters = classifier.best_params_

print(parameters)