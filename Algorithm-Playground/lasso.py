# Example of uisng Lasso Regression

# Import
from sklearn.linear_model import Lasso

# Create Lasso Regression
regression = Lasso(alpha = 0.1)

# Fit Lasso Regression
regression.fit([[0, 0], [1, 1]], [0, 1])

# Predict
prediction = regression.predict([[1, 1]])

# Get Regression Coefficients
coefficients = regression.coef_

# Output
print(prediction)
print(coefficients)