class Evaluation(object):
    # Constructor
    def __init__(self, emotion, tweet_set, precision=None, recall=None, fscore=None, accuracy=None):  #Constructor that initializes all the attributes
        self.emotion = emotion
        self.tweet_set = tweet_set
        self.precision = precision
        self.recall = recall
        self.fscore = fscore
        self.accuracy = accuracy

    # Methods
    def get_scores(self):                                                   #get_score method to calculate precision, recall, fscore and accuracy
        fp = 0
        fn = 0
        tn = 0
        tp = 0

        for t in self.tweet_set:
            if self.emotion in t.emotions and self.emotion in t.prediction:
                tp += 1
            elif self.emotion in t.emotions and self.emotion not in t.prediction:
                fn += 1
            elif self.emotion not in t.emotions and self.emotion in t.prediction:
                fp += 1
            elif self.emotion not in t.emotions and self.emotion not in t.prediction:
                tn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = (2 * precision * recall) / (precision + recall)
        accuracy = (tp + tn) / (tn + tp + fn + fp)

        self.precision = precision
        self.recall = recall
        self.fscore = fscore
        self.accuracy = accuracy
