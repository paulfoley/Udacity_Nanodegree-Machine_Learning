## Maximum Likelihood Hypothesis
# Find the maximum likelihood word based on the preceding word

sample_memo = '''
Milt, we're gonna need to go ahead and move you downstairs into storage B. We have some new people coming in, and we need all the space we can get. So if you could just go ahead and pack up your stuff and move it down there, that would be terrific, OK?
Oh, and remember: next Friday... is Hawaiian shirt day. So, you know, if you want to, go ahead and wear a Hawaiian shirt and jeans.
Oh, oh, and I almost forgot. Ahh, I'm also gonna need you to go ahead and come in on Sunday, too...
Hello Peter, whats happening? Ummm, I'm gonna need you to go ahead and come in tomorrow. So if you could be here around 9 that would be great, mmmk... oh oh! and I almost forgot ahh, I'm also gonna need you to go ahead and come in on Sunday too, kay. We ahh lost some people this week and ah, we sorta need to play catch up.
'''

# Takes in sample text and a word,
# and returns a dictionary with keys the set of words that come after, 
# whose values are the number of times the key comes after that word.
#   
# Just use .split() to split the sample_memo text into words separated by spaces.

def NextWordProbability(sampletext, word):
    # Create a List of Words
    words = sampletext.split()

    # Check if specific word is in word list
    if word in words:
    	indexes = [i for i, x in enumerate(words) if x == word]
    else:
    	return {}

    # The Indexes of the word after the preceding word
    indexes_next = [i+1 for i in indexes]

    # Return a list of the words after the proceding word
    word_list = [words[i] for i in indexes_next]
    word_count = {}

    for w in word_list:
    	if w in word_count:
    		word_count[w] += 1
    	else:
    		word_count[w] = 1

    return word_count


print(NextWordProbability(sample_memo, "Oh,"))