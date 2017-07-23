'''
Playing with Lasso Regression
'''

## Import Lasso
from sklearn.linear_model import Lasso

## Create Regression
regression = Lasso(alpha = 0.1)

## Fit Regression
regression.fit([[0, 0], [1, 1]], [0, 1])

## Predict
prediction = regression.predict([[1, 1]])

## Get Regression Coefficients
coefficients = regression.coef_

## Output
print(prediction)
print(coefficients)