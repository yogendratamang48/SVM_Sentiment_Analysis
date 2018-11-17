
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.models import load_model
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter
import string
import json
import pdb

DATASET = '../data/final_data_less.csv'
LSTM_MODEL_JSON = '../saved_model/model_lstm.json'
LSTM_MODEL_WEIGHTS = '../saved_model/model_lstm.h5'
HISTORY_FILE = '../saved_model/history_lstm.json'

# In[2]:

def save_lstm_model(model):
    # load json and create model
    model_json = model.to_json()
    with open(LSTM_MODEL_JSON, 'w') as jsonfile:
        jsonfile.write(model_json)
    # serialize weights to HDF5
    model.save_weights(LSTM_MODEL_WEIGHTS)

def load_lstm_model(model):
    # load weights into new model
    loaded_model = model_from_json(LSTM_MODEL_JSON)
    loaded_model.load_weights(LSTM_MODEL_WEIGHTS)
    # evaluate loaded model on test data
    return loaded_model


def get_train_test_words():
    '''
    converts to LSTM domain
    '''
    # global MAX_REVIEW_SIZE
    Sentiment = pd.read_csv(DATASET)
    #Join all the words in review to build a corpus
    all_text = ' '.join(Sentiment['clean_sentiment'])
    words = all_text.split()
    # Convert words to integers
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    sentiment_ints = []
    for each in Sentiment['clean_sentiment']:
        sentiment_ints.append([vocab_to_int[word] for word in each.split()])
    #Create a list of labels
    #labels = np.array([0 if each == 'negative' else 1 for each in Tweet['airline_sentiment'][:]])
    labels = Sentiment['sentiment'].values

    #Find the number of tweets with zero length after the data pre-processing
    sentiment_len = Counter([len(x) for x in sentiment_ints])

    seq_len = max(sentiment_len)
    features = np.zeros((len(sentiment_ints), seq_len), dtype=int)
    for i, row in enumerate(sentiment_ints):
        features[i, -len(row):] = np.array(row)[:seq_len]
    Y = pd.get_dummies(Sentiment['sentiment']).values
    split_frac = 0.8
    split_idx = int(len(features)*0.8)
    X = features
    # TRAIN_X, TEST_X = features[:split_idx], features[split_idx:]
    # TRAIN_Y, TEST_Y = Y[:split_idx], Y[split_idx:]
    train_x, test_x = features[0:split_idx], features[split_idx:]
    train_y, test_y = Y[0:split_idx], Y[split_idx:]
    return train_x, test_x, train_y, test_y, words

def TrainLSTM():
    train_x, test_x, train_y, test_y, words = get_train_test_words()
    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
          "\nValidation set: \t{}".format(test_x.shape))

    print("Train set: \t\t{}".format(train_y.shape),
          "\nTest set: \t\t{}".format(test_y.shape))

    embed_dim = 128
    lstm_out = 196
    max_features=len(words)

    model = Sequential()
    model.add(Embedding(max_features, embed_dim,input_length = train_x.shape[1]))
    model.add(Dropout(0.2, noise_shape=None, seed=None))
    model.add(LSTM(lstm_out, activation='tanh', recurrent_activation='hard_sigmoid', dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

    batch_size = 10
    history=model.fit(train_x, train_y, validation_split=0.2, epochs = 1, batch_size=batch_size, verbose=1)
    score,acc = model.evaluate(test_x, test_y, verbose = 2, batch_size = batch_size)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))

    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
    for x in range(len(test_x)):

        result = model.predict(test_x[x].reshape(1,test_x.shape[1]),batch_size=1,verbose = 2)[0]

        if np.argmax(result) == np.argmax(test_y[x]):
            if np.argmax(test_y[x]) == 0:
                neg_correct += 1
            else:
                pos_correct += 1

        if np.argmax(test_y[x]) == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1

    print("Positive Reviews: ", pos_cnt)
    print("Positive Correct: ", pos_correct)
    print("Negative Review Cont: ", neg_cnt)
    print("Negative Correct: ", neg_correct)
    print("pos_acc", pos_correct/pos_cnt*100, "%")
    print("neg_acc", neg_correct/neg_cnt*100, "%")
    print(history.history.keys())
    # Saving LSTM Model
    save_lstm_model(model)
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy/loss')
    plt.ylabel('accuracy-loss')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.savefig('../images/train_history.png')
    #saving history files
    with open(HISTORY_FILE, 'w') as history_file:
        json.dump(history.history, history_file)
    # summarize history for loss
if __name__=='__main__':
    TrainLSTM()
