## Example Using the sklearn MinMaxScaler 

# Imports
from sklearn.preprocessing import MinMaxScaler
import numpy as np

feature = np.array([[ 1., -1.,  2.],
					[ 2.,  0.,  0.],
					[ 0.,  1., -1.]])

# Rescale Data
feature_minmax = MinMaxScaler().fit_transform(feature)

# Output
print(feature_minmax)
