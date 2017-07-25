'''
Removitng outliers
'''
## Imports
import random
import numpy
import matplotlib.pyplot as plt
import pickle

## sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

## Function
def outlierCleaner(predictions, ages, net_worths):
    """
    Clean away the 10% of points that have the largest residual errors.
    (difference between the prediction and the actual net worth)

    Return a list of tuples named cleaned_data 
    where each tuple is of the form (age, net_worth, error).
    """
    cleaned_data = []
    errors = predictions - net_worths
    for i in range(0, len(predictions)):
        cleaned_data.append((ages[i], net_worths[i], errors[i]**2))
    
    cleaned_data = sorted(cleaned_data, key=lambda data: data[2])
    length = int(len(cleaned_data)- len(cleaned_data)*.1)
    
    return cleaned_data[:length]

## Load Up Data with Outliers in it
ages = pickle.load(open("practice_outliers_ages.pkl", "rb"))
net_worths = pickle.load(open("practice_outliers_net_worths.pkl", "rb"))

## Ages and net_worths need to be reshaped into 2D numpy arrays
ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))

## Create Training Features and Labels, as well as Test Features and Labels 
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

## Create Regression
reg = LinearRegression()

## Fit Regression with Outliers
reg.fit(ages_train, net_worths_train)
print('\nRegression Coefficient with Outliers:')
print(reg.coef_)

## Score Regression Wit Outliers
score = reg.score(ages_test, net_worths_test)
print('\nRegression Score with Outliers:')
print(score)


try:
    plt.plot(ages, reg.predict(ages), color="black")
except NameError:
    pass
plt.scatter(ages, net_worths)
plt.savefig('regregression-with_outliers.png', bbox_inches='tight')
plt.figure() # Reset Plot

## Identify and Remove the Most Outlier-y Points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner(predictions, ages_train, net_worths_train )
except NameError:
    print("your regression object doesn't exist, or isn't name reg")
    print("can't make predictions to use in identifying outliers")

# Only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

    # Refit your cleaned data!
    try:
        reg.fit(ages, net_worths)
        print('\nNew Regression Coefficient:')
        print(reg.coef_)

        score = reg.score(ages_test, net_worths_test)
        print('\nNew Regression Score:')
        print(score)

        plt.plot(ages, reg.predict(ages), color="blue")
    except NameError:
        print("you don't seem to have regression imported/created,")
        print("   or else your regression object isn't named reg")
        print("   either way, only draw the scatter plot of the cleaned data")
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.savefig('regregression-no_outliers.png', bbox_inches='tight')
else:
    print("outlierCleaner() is returning an empty list, no refitting to be done")

