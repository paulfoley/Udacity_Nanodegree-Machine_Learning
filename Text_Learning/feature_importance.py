# Reads in Features and Returns the Most Important Features

# Imports
# 
# Import
import pickle

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer

# The words (features) and authors (labels), already largely processed.
words_file = "/Users/Nexu/Desktop/Machine_Learning-Projects/Algorithms/text_learning/your_word_data.pkl" 
authors_file = "/Users/Nexu/Desktop/Machine_Learning-Projects/Algorithms/text_learning/your_email_authors.pkl"
word_data = pickle.load(open(words_file, "r"))
authors = pickle.load(open(authors_file, "r") )

'''
Create Training Features and Labels, as well as the Test Features and Lables
# test_size is the percentage of events assigned to the test set (the remainder go into training)
'''
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

# Create TfIdf Vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, 
                                max_df=0.5,
                                stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test)

'''
A classic way to overfit is to use a small number of data points and a large number of features;
Train on only 150 events to put ourselves in this regime
'''
features_train = features_train[:150]
labels_train   = labels_train[:150]

# Create Decision Tree Classifier
classifier = tree.DecisionTreeClassifier()

# Fit Classifier
classifier = classifier.fit(features_train, labels_train)

# Create Accuracy Score
score = classifier.score(features_test, labels_test)

# Show Feature Importance
'''
max_importance = max(classifier.feature_importances_)
index = classifier.feature_importances_.tolist().index(max_importance)
word = vectorizer.get_feature_names()[index])
print(word, max_importance)
'''

important_list = []
for feature in classifier.feature_importances_:
    if feature > .2:
        index = classifier.feature_importances_.tolist().index(feature)
        word = vectorizer.get_feature_names()[index]
        important_list.append([word, feature])

# Output
print(score)
print(important_list)
