##  Using a dense neural network to predict origin of a wine based on its description
##  Data file can be found at : https://www.kaggle.com/zynicide/wine-reviews

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')

# Read a csv into a pandas dataframe
def readData(path):
    return pd.read_csv(path, dtype=str)

# Only keep certain columns and drop all rows contaiing null
def cleanData(data, columnsToKeep):
    data = data[columnsToKeep]
    data = data.dropna(axis=0)
    return data

# Create a sequential model
# Add two dense layers
# Specify cross entropy loss and ADAM optimizer
# Fit the model to the training data
def trainModel(cv, Y, X_train, Y_train):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=len(cv.get_feature_names())))
    model.add(Dense(units=Y.max()+1, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=5, verbose=1, batch_size=32)
    return model

def main(_):
    PATH = 'winemag-data_first150k.csv'
    columnsToKeep = ['country', 'description']

    data = readData(PATH)
    data = cleanData(data, columnsToKeep)

    X = data['description'] # Input to the model
    Y = data['country']     # Output 

    ## Label Encoder - Gives each country a label number, US = 1, Spain = 2, etc
    labelEnc = LabelEncoder()
    Y = labelEnc.fit_transform(Y)

    ## Count Vectorizer - Tokenize a collection of text documents
    ## We build a vocabulary of words in the descriptions
    ## Lets us ignore case, punctuation and certain stopwords.
    cv = CountVectorizer(stop_words='english')
    X = cv.fit_transform(X)

    ## Split into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

    ## Create Model and train
    model = trainModel(cv, Y, X_train, Y_train)

    ## Evaluate model on test set
    result = model.evaluate(X_test, Y_test, verbose=1)
    print ('Accuracy : %s' % result[1])

if __name__ == '__main__':
    main(None)