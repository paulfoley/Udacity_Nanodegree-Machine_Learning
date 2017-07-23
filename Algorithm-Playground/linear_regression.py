'''
Playing with Linear Regression
'''

## Import Linear Regression
from sklearn.linear_model import LinearRegression

## Create Regression
regression = LinearRegression()

## Fit Regression
regression.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

## Output Coefficients
print(regression.coef_)