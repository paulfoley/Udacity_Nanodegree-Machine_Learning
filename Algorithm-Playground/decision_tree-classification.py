# Imports
from sklearn import tree
from sklearn.datasets import load_iris 
import IPython.display
import pydotplus 

iris = load_iris()

# Create Classifier
classifier = tree.DecisionTreeClassifier()

# Fit Classifier
classifier = classifier.fit(iris.data, iris.target)

# Create PDF of Decision Tree
dot_data = tree.export_graphviz(classifier, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("iris.pdf")  

# Predict Results
predictions = classifier.predict(iris.data[:1, :])

# Predict Probability Results
probability = classifier.predict_proba(iris.data[:1, :])

# Output Results
print(predictions, probability)