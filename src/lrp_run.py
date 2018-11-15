
import codecs
import numpy as np
from IPython.display import display, HTML
from LSTM.LSTM_bidi import *
from LSTM.LSTM_bidi import *
from util.heatmap import html_heatmap
import pandas as pd
import lstm_model

SEQUENCE_TEST = '../data/test_less.csv'


def get_test_sentence(sent_idx):
    """
    Returns a test set sentence and its label, sent_idx must be an integer in [1, 2210]"""
    df = pd.read_csv(SEQUENCE_TEST)
    print(df.shape)
    sentence = df['reviewText'][sent_idx]
    sentiment = df['sentiment'][sent_idx]
    sentence_ = lstm_model.sentiment_to_words(sentence)
    clean_words = sentence_.split()
    return clean_words, sentiment


def perform_lrp(words, target_class):
    '''
    performs lrp
    '''
    eps = 0.001
    # recommended value
    bias_factor = 0.0
    net = LSTM_bidi()
    # convert to word IDs
    w_indices = [net.voc.index(w) for w in words]
    Rx, Rx_rev, R_rest = net.lrp(
        w_indices, target_class, eps, bias_factor)  # LRP through the net
    # word relevances
    R_words = np.sum(Rx + Rx_rev, axis=1)
    scores = net.s.copy()
    print("prediction scores: ",   scores)
    print("\nLRP target class: ", target_class)
    print("\nLRP relevances:")
    for idx, w in enumerate(words):
        print("\t\t\t" + "{:8.2f}".format(R_words[idx]) + "\t" + w)
    print("\nLRP heatmap:")
    display(HTML(html_heatmap(words, R_words)))

    # sanity check
    bias_factor = 1.0                                    # value for sanity check
    Rx, Rx_rev, R_rest = net.lrp(w_indices, target_class, eps, bias_factor)
    R_tot = Rx.sum() + Rx_rev.sum() + R_rest.sum()  # sum of all "input" relevances

    print(R_tot)
    # check relevance conservation
    print(np.allclose(R_tot, net.s[target_class]))


if __name__ == '__main__':
    words, sentiment = get_test_sentence(16)
    perform_lrp(words, sentiment)
    print(words)
    print(sentiment)
