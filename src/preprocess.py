import pandas as pd
import json
import gzip
import code

COLUMNS_TO_KEEP = ['asin', 'overall', 'reviewText', 'summary']
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
    df = df[(df[column_name]>3)|(df[column_name]<3)]
    return df

if __name__=='__main__':
    # Loading dataset
    df = get_df_with_required_columns()
    print(df.shape)
    # Removing Netural Rows
    df = remove_neutral_rows(df)
    print(df.shape)
    # Remove NA values
    df = df.dropna(axis=0, how='any')
    print(df.shape)
    code.interact(local=locals())
