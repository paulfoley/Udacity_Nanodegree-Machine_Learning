'''
Use the sklearn dataset iris 
To show how the Decision Tree Classifier works
'''
## Imports
from subprocess import check_call

## sklearn imports
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_iris

## Get Data
iris = load_iris()

## Create Classifier
classifier = DecisionTreeClassifier()

## Fit Classifier
classifier = classifier.fit(iris.data, iris.target)

## Predict Results
prediction = classifier.predict(iris.data[:1, :])

## Output Results
print('Prediction:')
print(prediction)


## Create PNG of Decision Tree
dot_data = export_graphviz(classifier, out_file = 'iris.dot',feature_names = iris.feature_names, 
							class_names=iris.target_names, filled=True, rounded=True, special_characters=True)  

check_call(['dot','-Tpng','iris.dot','-o','iris.png'])
