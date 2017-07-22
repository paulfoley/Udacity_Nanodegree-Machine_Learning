## Bayes Optimal Classifier
# Compute the optimal label for a second missing word in a row
# based on the possible words that could be in the first blank

sample_memo = '''
Milt, we're gonna need to go ahead and move you downstairs into storage B. We have some new people coming in, and we need all the space we can get. So if you could just go ahead and pack up your stuff and move it down there, that would be terrific, OK?
Oh, and remember: next Friday... is Hawaiian shirt day. So, you know, if you want to, go ahead and wear a Hawaiian shirt and jeans.
Oh, oh, and I almost forgot. Ahh, I'm also gonna need you to go ahead and come in on Sunday, too...
Hello Peter, whats happening? Ummm, I'm gonna need you to go ahead and come in tomorrow. So if you could be here around 9 that would be great, mmmk... oh oh! and I almost forgot ahh, I'm also gonna need you to go ahead and come in on Sunday too, kay. We ahh lost some people this week and ah, we sorta need to play catch up.
'''

corrupted_memo = '''
Yeah, I'm gonna --- you to go ahead --- --- complain about this. Oh, and if you could --- --- and sit at the kids' table, that'd be --- 
'''

data_list = sample_memo.strip().split()

words_to_guess = ["ahead","could"]

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


def LaterWords(sample, word, distance):
    '''
    sample: a sample of text to draw from
    word: a word occuring before a corrupted sequence
    distance: how many words later to estimate (i.e. 1 for the next word, 2 for the word after that)
    
    returns: a single word which is the most likely possibility
    '''
    # Create a Dicttionary of Word Counts
    word_count = NextWordProbability(sample,word)
    
    for i in range(distance, distance + 1):
        word_dict = {}
        
        for w in word_count:
            temp_dict = NextWordProbability(sample, w)
            for w2 in temp_dict:
                word_dict[w2] =  temp_dict[w2] * word_count[w]

        word_count = word_dict

    sort_words = sorted(word_count.items(), key = lambda x:x[1],reverse = True)
    return sort_words[0][0]
    
print(LaterWords(sample_memo,"ahead",2))
