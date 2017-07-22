## Example of Random Forest

# Imports
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.ensemble import RandomForestClassifier

# Create Trainning Features and Labels, as well as Testing Features and Labels
features_train, labels_train, features_test, labels_test = makeTerrainData()

'''
The Training Data (features_train, labels_train) have both "fast" and "slow" points mixed together
Separate them so we can give them different colors in the scatterplot and identify them visually
'''
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


# Initial Visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()

############################# Random Forest ###########################
# Create Classifier
classifier = RandomForestClassifier()

# Fit Classifier
classifier.fit(features_train, labels_train)

# Predict
predictions = classifier.predict(features_test, labels_test)

# Accuracy Score
score = classifier.score(features_test, labels_test)

print(score)
prettyPicture(classifier, features_test, labels_test)
