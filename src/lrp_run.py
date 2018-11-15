from LSTM.LSTM_bidi import *
from LSTM.LSTM_bidi import * 
from util.heatmap import html_heatmap

import codecs
import numpy as np
from IPython.display import display, HTML


def get_test_sentence(sent_idx):
    """Returns a test set sentence and its label, sent_idx must be an integer in [1, 2210]"""
    idx = 1
    with codecs.open("./data/sequence_test.txt", 'r', encoding='utf8') as f:
        for line in f:
            line          = line.rstrip('\n')
            line          = line.split('\t')
            label         = int(line[0])-1         # true sentence class
            words         = line[1].split(' | ')   # sentence words
            if idx == sent_idx:
                return words, label
            idx +=1