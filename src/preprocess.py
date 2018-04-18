import pandas as pd
import json
import gzip
import code
import matplotlib.pyplot as plt
import numpy as np

COLUMNS_TO_KEEP = ['asin', 'overall', 'reviewText', 'summary']
FINAL_COLUMNS = ['sentiment', 'reviewText', 'word_count']
POSITIVES = 20000

def parse(path = None):
    '''
    parses gzip file
    '''
    if path is None:
        path='../data/reviews_Baby_5.json.gz'
    g = gzip.open(path, 'rb')
    for line in g:
        yield json.dumps(eval(line))

def get_dataframe(path=None):
    '''
    returns dataframe
    '''
    if path is None:
        path='../data/reviews_Baby_5.json.gz'
    i = 0
    df = { }
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def get_dataframe_from_json(path=None):
    '''
    returns dataframe from json file
    '''
    if path is None:
        path='../data/baby.json'
    df = pd.read_json(path)
    return df

def get_df_with_required_columns():
    '''
    return dataframe with required columns
    '''
    df = get_dataframe_from_json()
    # read from json file 
    df = get_dataframe_from_json()
    # Removing unwanted columns
    df = df[COLUMNS_TO_KEEP]
    # change empty or white space to NAN
    df['reviewText'] = df['reviewText'].apply(lambda x: x.strip()).replace('', np.nan)
    return df

def remove_neutral_rows(df, column_name=None, threshold=None):
    '''
    removes ratings of 3. they show more neutral sentiment
    We keep ratings less than 3 as negative
    We keep ratings greater than 3 as positive
    '''
    if column_name is None:
        column_name = 'overall'
    if threshold is None:
        threshold = 3
    df = df[(df[column_name]>4)|(df[column_name]<3)]
    return df


def add_sentiment_column(df):
    '''
    add 'sentiment' column from overall column
    '''
    df['sentiment']=df['overall'].apply(lambda x: 1 if x>4 else 0)
    df['word_count']=df['reviewText'].apply(lambda x: len(x.split(" ")))
    return df

def plot_data_distribution(plot_data):
    sentiments = list(plot_data.keys())
    values = list(plot_data.values())

    plt.bar(range(len(plot_data)), values, tick_label=sentiments)
    plt.ylabel("Count")
    plt.title("Sentiment Distribution")
    plt.savefig('../images/data_distribution.png')


def distribute_data(df):
    '''
    distributes dataframe into comparable number of classes
    '''
    df_positives = df[df['sentiment']==1].head(POSITIVES)
    df_negatives = df[df['sentiment']==0]

    positive_counts = len(df_positives)
    negative_counts = len(df_negatives)

    print('Negative Counts: ', negative_counts)
    print('Positive Counts: ', positive_counts)


    plot_data = {'positive':positive_counts, 'negative':negative_counts}

    df_combined=pd.concat([df_positives, df_negatives])
    # Randomize data
    df_combined = df_combined.sample(frac=1).reset_index(drop=True)

    # plot data distribution
    plot_data_distribution(plot_data)

    return df_combined

def get_roc_curve(model, X, y):
    '''
    Creating ROC Curve
    '''
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)
    return fpr, tpr
    
def plot_learning_curve(X, y, train_sizes, train_scores, test_scores, title='', ylim=None, figsize=(14,8)):
    '''
    Creating Learning Curve
    '''

    plt.figure(figsize=figsize)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="lower right")
    return plt

if __name__=='__main__':
    # Loading dataset
    df = get_df_with_required_columns()
    print(df.shape)
    # Removing Netural Rows
    df = remove_neutral_rows(df)
    print(df.shape)
    # Remove NA values
    df = df.dropna(axis=0, how='any')

    # Create Usable dataset conver ratings 4, 5 to sentiment 1
    # Ratings 1 and 2 to sentiment 0
    df = add_sentiment_column(df)
    df = distribute_data(df)
    # Write this to csv file
    df = df[FINAL_COLUMNS]
    df.to_csv('../data/final_data.csv', sep='\t')
    print(df.shape)
    code.interact(local=locals())
