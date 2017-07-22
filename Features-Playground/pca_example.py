# Imports
import numpy as np
from sklearn.decomposition import PCA

features = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

# Create PCA
pca = PCA(n_components=2)
pca.fit(features)

first_pc = pca.components_[0]
second_pc = pca.components_[1]

print(pca.explained_variance_ratio_) 