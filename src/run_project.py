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
import pickle

VARIABLES_PATH='../saved_variables/variables.pkl'
MODEL_PATH = '../saved_model/model.pkl'

def tokenize(text): 
    '''
    converts sentences into words list: tokenization
    '''
    return nltk.word_tokenize(text)

# Load trained model
def load_variables(path=None):
    '''
    loads variables in following order
    [X_train, y_train, X_test, y_test, train_scores, 
    test_scores, train_sizes,fpr, tpr]
    '''
    if path is None:
        path=VARIABLES_PATH
    with open(path, 'wb'):
        variables = pickle.load(path)
    return variables

def run_sentiment():
    '''
    loads trained model
    '''
    sa_model = joblib.load(MODEL_PATH)
    sentiment_string="TEST"
    print("For better result input review with more than 5 words: \n")
    
    while sentiment_string:
        sentiment_string = input("Enter Text to classify sentiment: \n")
        if sentiment_string.strip() != "":
            sentiment = sa_model.predict_proba([sentiment_string])
            print("SENTIMENT Information".center(30,"*"))
            print("POSITIVE Probability: %s" % sentiment[0][1])
            print("-".center(30,"-"))
            print("NEGATIVE Probability: %s" % sentiment[0][0])
            print("-".center(30,"-"))
            print("Final Sentiment".center(30, "*"))
            if sentiment[0][0]>sentiment[0][1]:
                print("NEGATIVE SENTIMENT")
            else:
                print("POSITIVE SENTIMENT:")
            print("=".center(50, "="))
        else:
            sentiment_string=None

if __name__=='__main__':
    run_sentiment()