# In this exercise we'll load the titanic data
# And then perform one-hot encoding on the feature names

# Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the dataset
data = pd.read_csv('titanic_data.csv')
# Limit to categorical data
data = data.select_dtypes(include=[object])

# Create a LabelEncoder object, which will turn all labels present in each feature to numbers. 
encoder_label = LabelEncoder()


# For each feature in X, apply the LabelEncoder's fit_transform function,
# Which will first learn the labels for the feature (fit)
# and then change the labels to numbers (transform). 

for feature in data:
    # use fit_transform on data[feature] using the LabelEncoder() object
    data[feature] = encoder_label.fit_transform(data[feature])

# Create a OneHotEncoder object, 
# which will create a feature for each label present in the data. 
encoder_hot = OneHotEncoder()

# Apply the OneHotEncoder's fit_transform function to all of data, 
# which will first learn of all the (now numerical) labels in the data (fit) 
# and then change the data to one-hot encoded entries (transform).
# Use fit_transform on X using the OneHotEncoder() object
encoded_label_data = encoder_hot.fit_transform(data)

print(encoded_label_data)