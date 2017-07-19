## K-means clustering on the Enron Data Set

# Python Imports
import pickle
import numpy
import matplotlib.pyplot as plt
import sys

# Skelarn Imports
from sklearn.cluster import KMeans

# Local File Imports
from feature_format import featureFormat, targetFeatureSplit
from draw_plot import Draw

# Load in the Dict of Dicts containing all the data on each person in the dataset
with open("dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Remove an outlier
data_dict.pop("TOTAL", 0)

# Find Max and Minimum Values of Stock Options

option_value_list = []
for data in data_dict:
    if data_dict[data]['exercised_stock_options'] != 'NaN':
        option_value_list.append(data_dict[data]['exercised_stock_options'])

print("Option Value, Max & Min:")
print(max(option_value_list))
print(min(option_value_list))


# Find Maximum and Minimum Values of Salary
salary_list = []
for data in data_dict:
    if data_dict[data]['salary'] != 'NaN':
        salary_list.append(data_dict[data]['salary'])

print("\nSalary Value, Max & Min:")
print(max(salary_list))
print(min(salary_list))

# The input features we want to use can be any key in the person-level dictionary 
# (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

# Plot features
for f1, f2, f3 in finance_features:
    plt.scatter(f1, f2)
plt.show()

# Cluster here; Create predictions of the cluster labels
# for the data and store them to a list called pred
kmeans = KMeans(n_clusters=2)
kmeans = kmeans.fit(finance_features)
pred = kmeans.predict(finance_features)

# Draw the Clusters
Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)