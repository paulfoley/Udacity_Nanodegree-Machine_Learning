## Example of Preprocessing Data using Sklearn

# Imports
import pandas
from sklearn.preprocessing import LabelEncoder

# Creating Sample Data
sample_data = {'name': ['Ray', 'Adam', 'Jason', 'Varun', 'Xiao'],
				'health': ['fit', 'slim', 'obese', 'fit', 'slim']}

# Storing Sample Data in the form of a Dataframe
data = pandas.DataFrame(sample_data, columns = ['name', 'health'])

# Create Label Encoder
label_encoder = LabelEncoder()

# Fit Label Encoder
label_encoder.fit(data['health'])

# Transform Data
data['health'] = label_encoder.transform(data['health'])

# Output Results
print(data)