from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import learning_curve
from sklearn.externals import joblib
import re

import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.models import load_model
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical

import lstm_model
# import pudb
# pudb.set_trace()

DATASET = '../data/final_data_less.csv'
MAX_REVIEW_SIZE = 555
LSTM_MODEL_JSON = '../saved_model/model_lstm.json'
LSTM_MODEL_WEIGHTS = '../saved_model/model_lstm.h5'
HISTORY_FILE = '../saved_model/history_lstm.json'


def load_lstm_model():
    # load weights into new model
    with open(LSTM_MODEL_JSON, 'r') as json_file:
        jsonData=json_file.read()
    loaded_model = model_from_json(jsonData)
    loaded_model.load_weights(LSTM_MODEL_WEIGHTS)
    # evaluate loaded model on test data
    return loaded_model

def get_train_test_words():
    '''
    converts to LSTM domain
    '''
    # global MAX_REVIEW_SIZE
    clean_sentiment = lstm_model.sentiment_to_words(raw_sentiment)
    sentiment_data = pd.read_csv(DATASET)
    Sentiment = sentiment_data.copy()
    #Pre-process the tweet and store in a
    Sentiment = sentiment_data.copy()
    #Pre-process the tweet and store in a separate column
    Sentiment['clean_sentiment']=Sentiment['reviewText'].apply(lambda x: lstm_model.sentiment_to_words(x))
    #Join all the words in review to build a corpus
    all_text = ' '.join(Sentiment['clean_sentiment'])
    words = all_text.split()
    # Convert words to integers
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    sentiment_ints = []

    sentiment_vector =[vocab_to_int[word] for word in clean_sentiment.split()]
    sentiment_ints.append(sentiment_vector)
    sentiment_len = Counter([len(x) for x in sentiment_ints])
    # seq_len = max(sentiment_len)
    seq_len = MAX_REVIEW_SIZE
    features = np.zeros((len(sentiment_ints), seq_len), dtype=int)
    for i, row in enumerate(sentiment_ints):
        features[i, -len(row):] = np.array(row)[:seq_len]
    return features

def return_train_test():
    '''
    returns train test
    '''

def lstm_probability(sentiment_string):
    """
    gets positive, negative probability of the sentiment from LSTM
    """
    features = convert_to_lstm_domain(sentiment_string)
    loaded_model  = load_lstm_model()
    sentiment = loaded_model.predict(features[0].reshape(1, MAX_REVIEW_SIZE),batch_size=1,verbose = 2)[0]
    positive = sentiment[1]
    negative = sentiment[0]
    return positive, negative

def classify():
    '''
    loads lstm model
    '''
    sentiment_string="TEST"
    while sentiment_string:
        sentiment_string = input("Enter Text to classify sentiment: \n")
        if sentiment_string.strip() != "":
            lstm_pos, lstm_neg = lstm_probability(sentiment_string)
            print("=".center(80, "="))
            print("Model".ljust(20)+"Positive Probability".ljust(30)+"Negative Probability".rjust(30))
            print("-".center(80, "-"))
            print("LSTM".ljust(20)+str(lstm_pos).center(30)+str(lstm_neg).rjust(30))
           
        else:
            sentiment_string=None


if __name__=='__main__':
    classify()
