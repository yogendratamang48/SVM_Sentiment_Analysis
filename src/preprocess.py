import pandas as pd
import json
import gzip
import code
import matplotlib.pyplot as plt
import numpy as np

COLUMNS_TO_KEEP = ['asin', 'overall', 'reviewText', 'summary']
FINAL_COLUMNS = ['sentiment', 'reviewText', 'word_count']
POSITIVES = 17500


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
df_final.to_csv('../data/final_data.csv')
df_final.to_json('../data/final_data.json')
print(df_final.shape)
#code.interact(local=locals())
