import pandas as pd
import json
import gzip
import code
import matplotlib.pyplot as plt
import numpy as np

COLUMNS_TO_KEEP = ['asin', 'overall', 'reviewText', 'summary']
FINAL_COLUMNS = ['sentiment', 'reviewText', 'word_count']
POSITIVES = 17500
TEST_SIZE = 1000


# Loading dataset
path='../data/baby.json'
df = pd.read_json(path, lines=True)
# Removing unwanted columns
df = df[COLUMNS_TO_KEEP]
# change empty or white space to NAN
df['reviewText'] = df['reviewText'].apply(lambda x: x.strip()).replace('', np.nan)
print(df.shape)
# Removing Netural Rows
df = df[(df['overall']>4)|(df['overall']<3)]
print(df.shape)
# Remove NA values
df = df.dropna(axis=0, how='any')
# Create Usable dataset conver ratings 4, 5 to sentiment 1
# Ratings 1 and 2 to sentiment 0
df['sentiment']=df['overall'].apply(lambda x: 1 if x>4 else 0)
df['word_count']=df['reviewText'].apply(lambda x: len(x.split(' ')))
df_positives = df[df['sentiment']==1].head(POSITIVES)
df_negatives = df[df['sentiment']==0]

positive_counts = len(df_positives)
negative_counts = len(df_negatives)

print('Negative Counts: ', negative_counts)
print('Positive Counts: ', positive_counts)


plot_data = {'positive':positive_counts, 'negative':negative_counts}

# Combining positive and negative reviews
df_combined=pd.concat([df_positives, df_negatives])

# Randomize/Shuffle data
df_combined = df_combined.sample(frac=1).reset_index(drop=True)

# plot data distribution
sentiments = list(plot_data.keys())
values = list(plot_data.values())

plt.bar(range(len(plot_data)), values, tick_label=sentiments)
plt.ylabel("Count")
plt.title("Sentiment Distribution")
plt.savefig('../images/data_distribution.png')
# Write this to csv file
df_final = df_combined[FINAL_COLUMNS]

df_final_less = df_final[:TEST_SIZE]

df_final.to_csv('../data/final_data.csv', index=False)
df_final_less.to_csv('../data/final_data_less.csv', index=False)

split_idx = int(df_final.shape[0]*0.8)
split_idx_ = int(df_final_less.shape[0]*0.8)

df_train, df_test = df_final[0:split_idx], df_final[split_idx:]
df_train_less, df_test_less = df_final_less[0:split_idx_], df_final_less[split_idx_:]

print(df_final.shape)
df_train.to_csv('../data/train.csv', index=False)
df_train_less.to_csv('../data/train_less.csv', index=False)

df_test.to_csv('../data/test.csv', index=False)
df_test_less.to_csv('../data/test_less.csv', index=False)
#code.interact(local=locals())
