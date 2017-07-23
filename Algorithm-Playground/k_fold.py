'''
Playing with K-Fold to Split Data into Training and Testing
'''

## Import K Fold
from sklearn.model_selection import KFold

## Create Features
features = ["a", "b", "c", "d"]

## K-Fold
kf = KFold(n_splits = 2)

## Output
for train, test in kf.split(features):
	print("%s %s " % (train, test))