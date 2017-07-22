# In this exercise we'll examine a learner which has high variance,
# and tries to learn nonexistant patterns in the data.
# Using the learning curve function from sklearn.learning_curve to plot learning curves
# of both training and testing error.

# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, learning_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score, make_scorer

# Set the learning curve parameters; you'll need this for learning_curves
size = 1000
cv = KFold(size, shuffle = True)
scorer = make_scorer(explained_variance_score)

# Create a series of data that forces a learner to have high variance
features = np.round(np.reshape(np.random.normal(scale=5,size=2*size),(-1,2)),2)
labels = np.array([[np.sin(x[0]+np.sin(x[1]))] for x in features])

def plot_curve():
    # Defining our regression algorithm
    reg = DecisionTreeRegressor()

    # Fit our model using X and y
    reg.fit(features, labels)

    # Score
    score = reg.score(features, labels)
    print ("Regressor score: " + str(score))
    
    # Use learning_curve imported above to create learning curves 
    # for both the training data and testing data.
    # You'll need reg, features, labels, cv and scorer from above.
    train_sizes, train_scores, test_scores = learning_curve(estimator = reg, X = features, y = labels, cv = cv)
    
    # Taking the mean of the test and training scores
    train_scores_mean = np.mean(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    
    # Plotting the training curves and the testing curves using train_scores_mean and test_scores_mean 
    plt.plot(train_sizes ,train_scores_mean,'-o',color='b',label="train_scores_mean")
    plt.plot(train_sizes,test_scores_mean ,'-o',color='r',label="test_scores_mean")
    
    # Plot aesthetics
    plt.ylim(-0.1, 1.1)
    plt.ylabel("Curve Score")
    plt.xlabel("Training Points")
    plt.legend(bbox_to_anchor=(1.1, 1.1))
    plt.show()

plot_curve()