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

def tokenize(text): 
    '''
    converts sentences into words list: tokenization
    '''
    return nltk.word_tokenize(text)

# Load trained model
MODEL_PATH = '../saved_model/model.pkl'
sa_model = joblib.load(MODEL_PATH)
sentiment_string = input("Enter Text to classify sentiment: \n")
sentiment = sa_model.predict([sentiment_string])
if sentiment[0] == [0]:
    print("NEGATIVE SENTIMENT")
else:
    print("POSITIVE SENTIMENT")