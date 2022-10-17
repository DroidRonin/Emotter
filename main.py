#The main file that binds all the modules together
from text_processing import *
from tweetclass import *
from perceptron import *
from mlp import *
from evaluation import *

# Labels
emotions = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']

# Train files
print('Creating train files.')
train_file = 'train.csv' 
train_list = csv2list(train_file)          #Retrieves the list with all the tweets in train file
train_tweets = get_tweets(train_list)       #Gets all the tweet instances in train file
train_vocab = vocabulary(train_list)        #Retrieves all the normalized vocabulary in train file

# Test files
print('Creating test files.')              
test_file = 'test.csv'
test_list = csv2list(test_file)                  #Retrieves the list with all the tweets in test file
test_tweets = get_tweets(test_list)              #Gets all the tweet instances in test file
test_vocab = vocabulary(test_list)              #Retrieves all the normalized vocabulary in test file

# Initialize perceptrons
print('Initializing perceptrons')
perceptrons = []                                 
for e in emotions:
    p = Perceptron(train_tweets, e)                          #Creates an instance of the Perceptron class
    perceptrons.append(p)

# Train perceptrons on tweets set
print('Training perceptrons.')
for p in perceptrons:
    p.training()                                            #This calls the training method of Perceptron class which updates the weights

# Predict all labels in each tweet in tweets set
print('Predicting labels.')
for t in train_tweets:
    mlp = MLP(t, perceptrons)
    mlp.predict_all()
    t.prediction = mlp.tweet.prediction

# Evaluation of our model
print('Evaluating model')
evaluations = []
for e in emotions:
    eval = Evaluation(e, train_tweets)
    evaluations.append(eval)

for eval in evaluations:
    eval.get_scores()
