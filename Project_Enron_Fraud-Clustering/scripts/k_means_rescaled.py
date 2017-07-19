## Example of Rescaling the data and then using k-means clustering

# Python Imports
import pickle
import numpy
import matplotlib.pyplot as plt
import sys

# Import sklearn Libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Import Local Files
from feature_format import featureFormat, targetFeatureSplit
from draw_plot import Draw

# Load in the Dict of Dicts Containing All the Data on Each Person in the Dataset
with open("dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Remove the TOTAL Outlier 
data_dict.pop("TOTAL", 0)

# Input features we want to use can be any key in the person-level dictionary 
# (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit(data)

# Print Out Specific Bonus and Stock Options Points
result = MinMaxScaler().fit(finance_features).transform([200000,1000000])
print(result)

# Rescale Features using MinMaxScaler
finance_features = MinMaxScaler().fit_transform(finance_features)

# Plot the Features
for f1, f2 in finance_features:
    plt.scatter(f1, f2)
plt.show()

# Create Kmeans Clusters
kmeans = KMeans(n_clusters=2)

# Fit Kmeans
kmeans = kmeans.fit(finance_features)

# Predict Kmeans
prediction = kmeans.predict(finance_features)

# Draw and Output PDF of Clusters
Draw(prediction, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)

