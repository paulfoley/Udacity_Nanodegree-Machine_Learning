# Project - Investigate Enron Fraud

In 2000, Enron was one of the largest companies in the United States. 
By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives. 


## Project Overview

In this project, we play detective for fun, and put our machine learning skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. The goal of this project is to investigate the Enron Fraud case and use machine learning to find the Persons of Interest (POIs). To do this we’ll be looking at financial and email data to see if we can predict the behaviors and indicators that would make someone a potential person of interest.


## Getting Started

### Prerequisites

You'll need to install:

* [Anaconda](https://www.continuum.io/downloads)
* [Python (Minimum 3)](https://www.continuum.io/blog/developer-blog/python-3-support-anaconda)
* [Pandas](https://anaconda.org/anaconda/pandas)
* [Numpy](https://anaconda.org/anaconda/numpy)
* [scikit-learn](https://anaconda.org/anaconda/scikit-learn)
* [Matplotlib](https://anaconda.org/anaconda/matplotlib)

### Data Files

* `dataset.pkl` - The dataset for the project, there are 146 people in the data set with 18 labeled as POI’s. From looking at the data, there are 1358 features that are missing values. The data has been wrangled and cleaned by [Udacity](https://www.udacity.com/).

#### Outliers

There are some people in the data set that don’t have real names. For our investigation purposes, we’ll remove these outliers with the name “TOTAL” and “THE TRAVEL AGENCY IN THE PARK “.


#### Features

There is a total of 19 Features that we can analyze for this project:

* Financial features: [`salary`, `deferral_payments`, `total_payments`, `loan_advances`, `bonus`, `restricted_stock_deferred`, `deferred_income`, `total_stock_value`, `expenses`, `exercised_stock_options`, `other`, `long_term_incentive`, `restricted_stock`, `director_fees`] All units are in US dollars.

* Email Features: [`to_messages`, `from_poi_to_this_person`, `from_messages`, `from_this_person_to_poi`, `shared_receipt_with_poi`] Units are generally number of emails messages.

#### Target Variable

* `poi` - Label of POI's in the dataset (boolean, represented as integer)


## Scripts

* `fraud_identification.py` - Main project script, reads in the data, takes the features of choice, and puts them into a numpy array, which is the input form that most [scikit-learn](https://anaconda.org/anaconda/scikit-learn) functions assume. The script then runs various machine learning algorithms (DecisitonTreeClassifier, SVM, Regression, and more) to identify POI's.
* `tester.py` - Python script that evaluates the analysis based on recall and precision scores.
`feature_format.py` - Python script that helps format the features used in the investigation.

### Output

The end goal for this program is finding a model that has greater then .3 recall and precision. 

Here are the results:

#### Decision Tree
* Score: 0.860465116279
* Recall: 0.4
* Precision: 0.4

#### Kmeans
* Score: -25.4062141797
* Recall: 0.277777777778
* Precision: 0.625

#### Naive Bayes
* Score: 0.860465116279
* Recall: 0.4
* Precision: 0.4

#### SVC
* Score: 0.883720930233
* Recall: 0.0
* Precision: 0.0

#### Logistic Regression
* Score: 0.883720930233
* Recall: 0.0
* Precision: 0.0

### Summary

The Gaussian Naïve Bayes model did the best job in optimizing for both recall and precision with the following values:
* Score = .86
* Recall = .4
* Precision = .4

A full write up can be found in the `Project Summary` word document.


## Authors

* **[Paul Foley](https://github.com/paulfoley)**
* [Udacity](https://www.udacity.com/)


## License

* <a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/"> Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>

<a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/">
	<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" />
</a>


## Acknowledgments

* [Udacity](https://www.udacity.com/)
