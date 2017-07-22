## Example of Linear Regression

# Import
from sklearn.linear_model import LinearRegression

# Create Regression
regression = LinearRegression()

# Fit Regression
regression.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

print(regression.coef_)