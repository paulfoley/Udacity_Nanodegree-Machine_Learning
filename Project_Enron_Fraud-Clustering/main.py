## Idenity Persons of Interest (Poi's) in the Enron Fraud Case using Machine Learning

# Python Imports
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Import sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, f_classif
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Import From Local Files
from tester import dump_classifier_and_data
from feature_format import featureFormat, targetFeatureSplit

#################### Analyze Features ####################
'''
Select the features to analyze.
features_list is a list of strings, each of which is a feature name.
(The first feature must be "poi")
'''
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

# Load the dictionary containing the dataset
with open("dataset.pkl", "r") as data_file:
	data_dict = pickle.load(data_file)

# Output Number of People We're going to analyze
print('Number of People To Investigate:')
print(len(data_dict))

# Output # of POI's
num_poi = 0
for data in data_dict:
	if data_dict[data]['poi'] == 1:
		num_poi += 1
print('\nNumber of Labeled POIs:')
print(num_poi)

# Of Features with NaN
num_nan = 0
for data in data_dict:
	for person in data_dict[data]:
		if data_dict[data][person] == 'NaN':
			num_nan +=1

print('\nNumber of Features Missing Values:')
print(num_nan)


#################### Remove outliers ####################

# Remove Fake People
data_dict.pop('TOTAL',)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',)


# Find other Outliers based on Salary and Bonus
print("\nOutliers Based on Salary and Bonus:")
for data in data_dict:
	if data_dict[data]['salary'] > 1000000 and data_dict[data]['salary'] != 'NaN':
		if data_dict[data]['bonus'] > 5000000 and data_dict[data]['bonus'] != 'NaN':
			print(data)

# This function returns Jeff Skilling and Ken Lay, the two top dogs,
# Their salaries and bonuses definitely deviate from the norm


#################### Feature Creation ####################

# Let's Calculate the Fraction of From and To POI's as new features
def computeFraction(poi_messages, all_messages):
    """ 
    Given a number messages to/from POI (numerator) 
    and number of all messages to/from a person (denominator),
    Return the fraction of messages to/from that person that are from/to a POI.
    
    When "NaN", there is no known email address
    (and so no filled email features), and integer division!
    In case of poi_messages or all_messages having "NaN" value,
    the program will return 0.
    """
    fraction = 0.
    if poi_messages == "NaN":
        fraction = 0.
    elif all_messages == "NaN":
        fraction = 0.
    else: 
        fraction = float(poi_messages)/float(all_messages)

    return fraction

for name in data_dict:
	data_point = data_dict[name]

	# Calculate Fraction From POI
	from_poi_to_this_person = data_point["from_poi_to_this_person"]
	to_messages = data_point["to_messages"]
	fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
	data_dict[name]["fraction_from_poi"] = fraction_from_poi

	# Calculate Fraction To POI
	from_this_person_to_poi = data_point["from_this_person_to_poi"]
	from_messages = data_point["from_messages"]
	fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
	data_dict[name]["fraction_to_poi"] = fraction_to_poi

# Add the New Fraction Features to the Features List
features_list.append("fraction_from_poi")
features_list.append("fraction_to_poi")

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Scaling Features
features = MinMaxScaler().fit_transform(features)

# Create Training Features and Labels, as well as Testing Features and Labels
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# To Speed Up Testing We'll Use a Smaller Data Set
features_train = features_train[:150]
labels_train = labels_train[:150]

# Use SelectKBest to get the top features
k_scorer = SelectKBest(score_func = f_classif, k="all")
scorer = k_scorer.fit(features_train, labels_train)
fit_scores = sorted(scorer.scores_, reverse=True)

y_pos = np.arange(len(features_list[1:]))
plt.bar(y_pos, fit_scores, align='center', alpha=0.5)
plt.xticks(y_pos, features_list[1:], rotation=90)
plt.xlabel('Features')
plt.ylabel('Feature Score')
plt.title('Features and Score') 
plt.show()

# Number of Features with a scor over 5:
print('\nNumber of Features With a Score Over 5:')
count = 0
for score in fit_scores:
	if score > 5:
		count += 1
print(count)

# From the Plot we see the varying scores for each feature. 
# Using the cutoff technique we'll say features that have scores greater then 5 we'll keep.
# That gives us a k value of 10

k_best = SelectKBest(score_func = f_classif, k=11)
fitter = k_best.fit(features_train, labels_train)
features_train = fitter.transform(features_train)
features_test = fitter.transform(features_test)
features = fitter.transform(features)


#################### Create Classifiers ####################
'''
Try a varity of classifiers

For more info: http://scikit-learn.org/stable/modules/pipeline.html
'''
########## Decision Tree Classifier ##########

clf_dt = DecisionTreeClassifier()
clf_dt = clf_dt.fit(features_train, labels_train)

########## K Means Clustering ##########

clf_kmeans = KMeans(n_clusters=2)
clf_kmeans = clf_kmeans.fit(features)

########## Naive Bayes ##########
clf_gnb = GaussianNB()
clf_gnb = clf_gnb.fit(features_train, labels_train)

########## SVC ##########
clf_svc = SVC(C = 100.0, kernel = 'rbf', gamma = .001, probability = False)
clf_svc = clf_svc.fit(features_train, labels_train)

########## Logistic Regression ##########
# Use Grid Search CV to Optimize Parameters for Logistic
n_components = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Cs = np.logspace(-4, 4, 3)
pipe_log = Pipeline(steps=[('pca', PCA()), ('logistic', LogisticRegression())])
clf_log = GridSearchCV(pipe_log, dict(pca__n_components=n_components, logistic__C=Cs))
clf_log = clf_log.fit(features_train, labels_train)


#################### Classifier Scores ####################
'''
Tune the classifier to achieve better than .3 precision and recall using the testing script.
Check the tester.py script in the project folder for details on the evaluation method,
especially the test_classifier function.
Because of the small size of the dataset,
the script uses stratified shuffle split cross validation. 

For more info: 
http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
'''
print("\nScores From Various Machine Learning Models:")

########## Decision Tree Classifier ##########
# Create Predcitions
predict_dt = clf_dt.predict(features_test, labels_test)

# Score, Recall & Precision
score_dt = clf_dt.score(features_test, labels_test)
recall_dt = recall_score(labels_test, predict_dt)
precision_dt = precision_score(labels_test, predict_dt)

# Output
print("Decision Tree - Score, Recall, and Precision:")
print(score_dt)
print(recall_dt)
print(precision_dt)

########## K Means Clustering ##########
# Create Predcitions
predict_kmeans = clf_kmeans.predict(features)

# Score, Recall & Precision
score_kmeans = clf_kmeans.score(features, labels)
recall_kmeans = recall_score(labels, predict_kmeans)
precision_kmeans = precision_score(labels, predict_kmeans)

# Output
print("\nKmeans - Score, Recall, and Precision:")
print(score_kmeans)
print(recall_kmeans)
print(precision_kmeans)

########## Naive Bayes ##########
# Create Predcitions
predict_gnb = clf_gnb.predict(features_test)

# Score, Recall & Precision
score_gnb = clf_gnb.score(features_test, labels_test)
recall_gnb = recall_score(labels_test, predict_gnb)
precision_gnb = precision_score(labels_test, predict_gnb)

# Output
print("\nNaive Bayes - Score, Recall, and Precision:")
print(score_gnb)
print(recall_gnb)
print(precision_gnb)

########## SVC ##########
# Create Predcitions
predict_svc = clf_svc.predict(features_test)

# Score, Recall & Precision
score_svc = clf_svc.score(features_test, labels_test)
recall_svc = recall_score(labels_test, predict_svc)
precision_svc = precision_score(labels_test, predict_svc)

# Output
print("\nSVC - Score, Recall, and Precision:")
print(score_svc)
print(recall_svc)
print(precision_svc)

########## Logistic Regression ##########
# Create Predcitions
predict_log = clf_log.predict(features_test)

# Score, Recall & Precision
score_log = clf_log.score(features_test, labels_test)
recall_log = recall_score(labels_test, predict_log)
precision_log = precision_score(labels_test, predict_log)

# Output
print("\nLogistic Regression - Score, Recall, and Precision:")
print(score_log)
print(recall_log)
print(precision_log)


#################### Tester ####################
'''
Dump the best classifier, dataset, and features_list so results can be checked in tester.py. 
'''
dump_classifier_and_data(clf_gnb, my_dataset, features_list)
