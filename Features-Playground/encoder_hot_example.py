## Example of a Hot Encoder

# Imports
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# Creaet sample data
sample_data = {'name': ['Ray', 'Adam', 'Jason', 'Varun', 'Xiao'],
				'health':['fit', 'slim', 'obese', 'fit', 'slim']}

# Store Sample Data in the form of a Dataframe
data = pd.DataFrame(sample_data, columns = ['name', 'health'])
pandas.get_dummies(data['health'])

# Create Label Encoder
encoder_label = LabelEncoder()
encoded_label_data = encoder_label.fit_transform(data['health'])

# Create OneHotEncoder Object
encoder_hot = OneHotEncoder()
results = encoder_hot.fit_transform(encoded_label_data.reshape(-1, 1))

print(results)