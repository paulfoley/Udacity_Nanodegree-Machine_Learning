# Get Stopwords from NLTK
import nltk
nltk.download()

sw = nltk.corpus.stopwords.words("english")

print(len(sw))