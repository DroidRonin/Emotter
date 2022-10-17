from text_processing import normalize


class Tweet(object):                    #Creating Tweet Class
    # Constructor
    def __init__(self, emotions=None, prediction=None, text=None, features=None):
        # List of true labels
        self.emotions = emotions
        # List of predicted labels
        self.prediction = prediction
        # String of text in tweet
        self.text = text
        # Normalised word types in tweet
        self.features = features

    #Feature Extraction Method that returns all the normalized features
    def feature_extraction(self):
        self.features = []
        tokens = normalize(self.text)
        for w in tokens:
            if w not in self.features:
                self.features.append(w)
        return self.features


# Create list of tweet instances having attributes - 'emotions', 'prediction', 'text', 'features'
def get_tweets(tweet_list):
    tweet_instances = []
    for t in tweet_list:
        tweet = Tweet()
        tweet.emotions = t[:8]
        tweet.prediction = []
        tweet.text = t[-1]
        tweet.features = tweet.feature_extraction()
        tweet_instances.append(tweet)
    return tweet_instances
