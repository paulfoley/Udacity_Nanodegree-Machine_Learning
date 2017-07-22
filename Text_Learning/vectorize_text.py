"""
    Code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""

# Imports
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle
import re
import sys
sys.path.append( "/Users/Nexu/Desktop/Machine_Learning-Projects/Project_Enron/tools/" )
from parse_out_email_text import parseOutText

# Emails From
from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")
from_data = []
word_data = []


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        # Only look at first 200 emails when developing
        # Once everything is working, remove this line to run over full dataset
        path = os.path.join('/Users/Nexu/Desktop/Machine_Learning-Projects/Project_Enron/', path[:-1])
        email = open(path, "r")

        # Use parseOutText to extract the text from the opened email            
        email_text = parseOutText(email)

        # Use .replace() to take out "replace_words"
        replace_words = ["sara", "shackleton", "sshacklensf", "chris", "germani", "cgermannsf"]
        for word in replace_words:
            email_text = email_text.replace(word, '')

        # Append the text to word_data
        word_data.append(email_text)
            
        # Create from_data
        if name == 'sara':
            # Append a 0 to from_data if email is from Sar
            from_data.append(0)
        elif name == 'chris':
            # Append a 1 if email is from Chris
            from_data.append(1)

        email.close()

from_sara.close()
from_chris.close()

pickle.dump(word_data, open("your_word_data.pkl", "w") )
pickle.dump(from_data, open("your_email_authors.pkl", "w") )

# print(word_data[152])

# Create TfidfVectorizer and Fit word_data
vectorizer = TfidfVectorizer(stop_words = 'english', lowercase = True)
vectorizer.fit_transform(word_data)

#feature_names = vectorizer.get_feature_names()
#print(len(feature_names))

feature_name = vectorizer.get_feature_names()[34597]
print(feature_name)
