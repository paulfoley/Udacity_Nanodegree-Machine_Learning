# Project Introduction
In 2000, Enron was one of the largest companies in the United States. 
By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. 
In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives. 

In this project, we play detective for fun, and put our machine learning skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal.

## Resources Needed
Python and sklearn running on your computer, as well as the starter code (both python scripts and the Enron dataset) that you can download.

## Some relevant files: 
main.py : Main python script, run this to run the program

dataset.pkl : The dataset for the project

tester.py : Evaluates the analysis

## Steps to Success
'main.py' reads in the data, takes the features of choice, and puts them into a numpy array, which is the input form that most sklearn functions assume. The script then runs various machine learning algorithms (DecisitonTreeClassifier, SVM, Regression, and more) to identify POI's.

The features in the data fall into three major types, namely financial features, email features, and POI labels.

financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

POI label: [‘poi’] (boolean, represented as integer)

## Project Write Up
A summary of the project and the findings can be found in the "Project_Summary.docx" file
