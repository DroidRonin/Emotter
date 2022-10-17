from nltk.corpus import stopwords
import re

# emotions = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']


def normalize(text):
    # Remove unnecessary characters
    cleanup = re.sub('(@\w+|\d+|[><+=*$%/.,\-\'\"]+)', ' ', text)
    cleanup = re.sub('!', ' !', cleanup)
    # Handle hashtags
    no_hashtag = []
    hashtags = re.findall('#\w+', cleanup)
    final = re.sub('#\w+', '', cleanup)
    tokens = final.split()
    for h in hashtags:
        if h.islower():
            h = re.sub('#', '', h)
            no_hashtag.append(h)
        else:
            words = re.findall('[A-Z][^A-Z]*', h)
            for w in words:
                no_hashtag.append(w)
    for el in no_hashtag:
        tokens.append(el)
    # Lowercase all words
    lowercase = [w.lower() for w in tokens]
    # Remove stopwords
    sw = stopwords.words("english")
    for w in lowercase:
        if w in sw:
            lowercase.remove(w)
    return lowercase


# Create list with tweets
def csv2list(csvfile):
    openfile = open(csvfile)
    csvfile = openfile.readlines()

    tweets = []
    for row in csvfile:
        columns = row.split(sep='\t')
        tweets.append(columns)
    # for t in tweets:
        # t[-1] = normalize(t)
    return tweets

#This retrieves the tweets returns normalized tokens
def vocabulary(tweet_list):
    vocab = []
    for t in tweet_list:
        text = normalize(t[-1])
        for w in text:
            if w not in vocab:
                vocab.append(w)
    return vocab
