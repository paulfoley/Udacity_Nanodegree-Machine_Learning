"""
Facial recognition using eigenfaces and SVMs

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild" (LFW):
* http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
* http://vis-www.cs.umass.edu/lfw/
* http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html
"""
## Imports
from time import time
import logging
import pylab as pl
import pandas as pd
import numpy as np

## sklearn Imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

## Display progress logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

## Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

## Introspect the Images Arrays to Find the Shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
np.random.seed(42)

## Use the data directly (as relative pixel position info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

## The Label to Predict is the Id of the Person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

## Output Data Metrics
print("\nTotal dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

## Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

## Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled dataset)
## Unsupervised feature extraction / dimensionality reduction
n_components = 50
print("\nExtracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(svd_solver='randomized', n_components=n_components, whiten=True).fit(X_train)
print("Done in %0.3fs" % (time() - t0))

## Explained Variance Ratio's
print("\nFind the First and Second Variance Ratios")
ratios = pca.explained_variance_ratio_
first_pc = ratios[0]
second_pc = ratios[1]
print("First Variance Ratio:")
print(first_pc)
print("Second Variance Ratio:")
print(second_pc)

## Eigenfaces
eigenfaces = pca.components_.reshape((n_components, h, w))
print("\nProjecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("Done in %0.3fs" % (time() - t0))

## Train a SVM classification model
print("\nFitting the classifier to the training set")
t0 = time()
param_grid = {
              'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
              }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("Done in %0.3fs" % (time() - t0))

## Grid Search Parameters
print("\nBest parameters found by grid search:")
parameters = clf.best_params_
print(parameters)
print("\nBest estimator found by grid search:")
estimators = clf.best_estimator_
print(estimators)

## Quantitative Evaluation of the Model Quality
print("\nPredicting the people names on the testing set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("Done in %0.3fs" % (time() - t0))

def classifaction_report_csv(report):
    ### Helper function output classification report as a CSV
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split(' ')
        clean_row_data = []
        for r in row_data:
            if r != '' and r != 'W':
                clean_row_data.append(r)
        row['First Name'] = clean_row_data[0]
        row['Last Name'] = clean_row_data[1]
        row['precision'] = float(clean_row_data[2])
        row['recall'] = float(clean_row_data[3])
        row['f1_score'] = float(clean_row_data[4])
        row['support'] = float(clean_row_data[5])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report.csv', index = False)

report = classification_report(y_test, y_pred, target_names=target_names)
classifaction_report_csv(report)

matrix = confusion_matrix(y_test, y_pred, labels=range(n_classes))
np.savetxt('confusion_matrix.csv', matrix, delimiter=",", fmt='%10.5f')

## Qualitative Evaluation of the Predictions
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    # Helper function to plot a gallery of portraits
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())

## Plot the Result of the Prediction on a Portion of the Test Set
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
plot_gallery(X_test, prediction_titles, h, w)

## Plot the Gallery of the Most sSgnificative Eigenfaces
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
pl.show()
