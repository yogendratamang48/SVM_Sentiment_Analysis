## Prelimininary Steps
* Download dataset from [amazon dataset page](http://jmcauley.ucsd.edu/data/amazon/)
* unzip the `reviews_Baby_5.json.gz` file and rename the json file to
  `baby.json` ( your wish)
* Make the json file more like list of json. For this I simply did:
** `sed -i 's|$|,|g` - adds comma(`,`) at each end of line
** add `[` to first character of the file
** replace `,` with `]` at last character of file
```python
def get_dataframe_from_json(path=None):
    '''
    returns dataframe from json file
    '''
    if path is None:
        path='../data/baby.json'
    df = pd.read_json(path)
    return df
```
* Now you have dataframe
## Data Preprocess steps:
### Preparing Datasets
- `overall` column gives user's rating for given product. This columns value range is upto 5
- We convert rating greater than 3 to Positive Sentiment
- We convert rating less than 3 to Negative Sentiment
#### Data Cleaning Removing reviews with empty string
### Data Distribution
- We are using 20,000 reviews of positive sentiments
- We are using 17001 reviews of negative sentiments
![data distribution]('../images/data_distribution.png') 

### 
