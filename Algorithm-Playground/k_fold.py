## Example of using K-Fold to Split Data into Training and Testing

# Imports
from sklearn.model_selection import KFold

features = ["a", "b", "c", "d"]
kf = KFold(n_splits = 2)
for train, test in kf.split(features):
	print("%s %s " % (train, test))