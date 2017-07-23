## Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## sklearn Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve

## Get Data
data = pd.read_csv('data.csv')
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

## Randomize Data
np.random.seed(55)

def randomize(X, Y):
    permutation = np.random.permutation(Y.shape[0])
    X2 = X[permutation,:]
    Y2 = Y[permutation]
    return X2, Y2

X2, y2 = randomize(X, y)


## Estimators
### Logistic Regression
estimator_log = LogisticRegression()

### Decision Tree
estimator_gbc = GradientBoostingClassifier()

### Support Vector Machine
estimator_svc = SVC(kernel='rbf', gamma=1000)

## Learning Curve Function
def draw_learning_curves(X, y, estimator, num_trainings):
	### Draws the Learning Curve given an estimator

	#### Get the Training Sizes, Train Scores, and Test Scores
    train_sizes, train_scores, test_scores = learning_curve(estimator, X2, y2, 
    														cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    #### Plot the Learning Curve
    plt.grid()
    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.plot(train_scores_mean, 'o-', color="g",
             label="Training score")
    plt.plot(test_scores_mean, 'o-', color="y",
             label="Cross-validation score")
    plt.legend(loc="best")

    plt.show()

## Output
draw_learning_curves(X, y, estimator_log, 5)
draw_learning_curves(X, y, estimator_gbc, 5)
draw_learning_curves(X, y, estimator_svc, 5)
