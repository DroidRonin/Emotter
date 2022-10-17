class MLP(object):
    def __init__(self, tweet, perceptrons):     
        self.tweet = tweet
        self.perceptrons = perceptrons

    # Methods
    def predict_all(self):
        # Predict all emotions (updates tweet.prediction attribute)
        tweet = self.tweet
        features = tweet.features
        tweet.prediction = []
        perceptrons = self.perceptrons

        def tweet_sum(weights):
            # Input: Dictionary of weights for one emotion
            weight_sum = 0
            for f in features:
                if f in weights:
                    weight_sum += weights[f]
            # Output: Sum of the weights of features found in the tweet
            return weight_sum

        for p in perceptrons:
            e = p.emotion
            w = p.weights
            bias = p.bias
            t_sum = tweet_sum(w)
            if t_sum >= bias:            #This checks if the sum of weights > and takes treats that as a valid positive prediction
                tweet.prediction.append(e)    
