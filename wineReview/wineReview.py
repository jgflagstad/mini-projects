## Messing around with a wine review data file
## Data file can be found at : https://www.kaggle.com/zynicide/wine-reviews

## TODO: 
##  Only keep adjectives?
##  Predict which country based on review?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from collections import Counter
PATH = 'winemag-data_first150k.csv'

data = pd.read_csv(PATH)

## Data cleaning
## Only keeping countries with more than 1000 wines
## Combining all descriptions of wines from a country into one large feature
countedCountries = data.groupby('country')['description'].nunique().sort_values(ascending=False).reset_index(name='count')
countriesToKeep = countedCountries[countedCountries['count'] > 1000]
countriesToKeepList = countriesToKeep['country'].tolist()
joinedData = data.groupby('country')['description'].apply(' '.join).reset_index()
joinedDataWithKeptCountries = joinedData[joinedData['country'].isin(countriesToKeepList)]

# Ignoring boring words
ignoredWords = ['and', 'the', 'a', 'of', 'with', 'is', 'in', 'this', 'to', 'or', 'where', 'it', 'wine', 'but', 
                'that', 'for', 'on', 'from', 'are', 'has', 'its', 'an', 'as', 'flavors', 'made', 'by', 'leads', 
                'into', 'at', 'drink', 'hints', 'while', 'finish', 'more', 'than', 'ways', 'one', 'between',
                'made', 'wine', 'not', 'some', 'offers', 'now', 'slightly', 'very', 'up', 'well', 'very', 'shows']

commonWords = []

for index, row in joinedDataWithKeptCountries.iterrows():
    descriptions = row['description'].split()
    # Removing punctuation and higher case letters
    removingPunc = [''.join(c.lower() for c in s if c not in string.punctuation) for s in descriptions]
    DescCounter = Counter(removingPunc)
    # Removing words found in ignoredWords
    for word in list(DescCounter):
        if word in ignoredWords:
            del DescCounter[word]
    most_occur = DescCounter.most_common(20)
    commonWords.append([row['country'], ': ', [element[0] for element in most_occur]])

print('\n'.join(' '.join(map(str,element)) for element in commonWords))
