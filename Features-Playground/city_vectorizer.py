from sklearn.feature_extraction import DictVectorizer

measurements = [{'city': 'Dubai', 'temperature': 33.},
				{'city': 'London', 'temperature': 12.},
				{'city': 'San Fransisco', 'temperature': 18.}]

vectorizer = DictVectorizer()
arr = vectorizer.fit_transform(measurements).toarray()
names = vectorizer.get_feature_names()

print(arr)
print(names)