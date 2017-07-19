## Look for Outliers in the Enron Data Set

# Python Imports
import pickle
import matplotlib.pyplot
import sys

# Import Local Files
from feature_format import featureFormat, targetFeatureSplit

# Read in Data Dictionary, Convert to Numpy Array
data_dict = pickle.load(open("dataset.pkl", "r"))
data_dict.pop('TOTAL',)

# Print Outliers that have both unusually high Salaries and Bonuses
for data in data_dict:
	if data_dict[data]['salary'] > 1000000 and data_dict[data]['salary'] != 'NaN':
		if data_dict[data]['bonus'] > 5000000 and data_dict[data]['bonus'] != 'NaN':
			# Print Outliers
			print(data)

# Display a Scatter Plot of Individuals Salary and Bonuses
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
