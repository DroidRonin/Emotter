from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np


# Retrieve data sets
def load_data(filepaths, df_list):
    for data, path in filepaths.items():
        df = pd.read_csv(path, sep='\t', encoding='ansi')
        df_list.append(df)


# Load training data
training_paths = {'training': 'D:/MSc Computational Linguistics/TeamLab_NLP/Data/Sentiment/'
                              'data-all-annotations/trainingdata-all-annotations.txt',
                  'trial': 'D:/MSc Computational Linguistics/TeamLab_NLP/Data/Sentiment/'
                           'data-all-annotations/trialdata-all-annotations.txt'}
training_list = []
load_data(training_paths, training_list)
training_data = pd.concat(training_list)

# Tweet texts for training
training_tweets = training_data['Tweet'].values

# Load test data
test_paths = {'taskA': 'D:/MSc Computational Linguistics/TeamLab_NLP/Data/Sentiment/'
                       'data-all-annotations/testdata-taskA-all-annotations.txt',
              'taskB': 'D:/MSc Computational Linguistics/TeamLab_NLP/Data/Sentiment/'
                       'data-all-annotations/testdata-taskB-all-annotations.txt'}
test_list = []
load_data(test_paths, test_list)
test_data = pd.concat(test_list)

# Tweet texts for testing
test_tweets = test_data['Tweet'].values

# Text pre-processing
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(training_tweets)

X_train = tokenizer.texts_to_sequences(training_tweets)
X_test = tokenizer.texts_to_sequences(test_tweets)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# Word embeddings using GloVe
def create_embedding_matrix(filepath, word_i, emb_dim):
    vocab_size = len(word_i) + 1  # Adding 1 because of reserved 0 index
    emb_matrix = np.zeros((vocab_size, emb_dim))

    with open(filepath, encoding='utf8') as f:
        for line in f:
            word, *vector = line.split()
            if word in word_i:
                idx = word_i[word]
                emb_matrix[idx] = np.array(vector, dtype=np.float32)[:emb_dim]

    return emb_matrix


embedding_dim = 100
embedding_matrix = create_embedding_matrix('D:/MSc Computational Linguistics/TeamLab_NLP/Data/glove.twitter.27B.100d.txt',
                                           tokenizer.word_index, embedding_dim)

# Check how much of the vocabulary is covered by the pre-trained model
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print(nonzero_elements / vocab_size)  # 0.72

"""Work in progress beneath this point"""

# Model configuration
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# Evaluation
