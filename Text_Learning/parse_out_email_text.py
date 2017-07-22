# Import Stemmer from NLTK
from nltk.stem.snowball import SnowballStemmer
import string

stemmer = SnowballStemmer('english')

def parseOutText(f):
    """ 
        Given an opened email file f, parse out all text below the metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
    """

    # Go back to beginning of file
    f.seek(0)  
    all_text = f.read()

    # Split Off Metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        # Remove Punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        # Split the text string into individual words, stem each word
        word_list = text_string.split()
        for word in word_list:
            # Make sure there's a single space between each stemmed word     
            # Append the stemmed word to words
            words += stemmer.stem(word) + " "

    return words

def main():
    ff = open("test_email.txt", "r")
    text = parseOutText(ff)
    print(text)

main()

