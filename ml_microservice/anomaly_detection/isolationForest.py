import pandas as pd

import os
import configparser

conf = configparser.ConfigParser()
conf.read('config.ini')

csv_file = os.path.join(conf['Resources']['datasetsPath'], 'jena_climate_2009_2016.csv')
jena_df = pd.read_csv(csv_file)
T = jena_df['T (degC)']

# preprocessing
def split_standardize_jena(T):
    n = len(T)
    train, dev, test = T[:int(n*.7)], T[int(n*.7):int(n*.9)], T[int(n*.9):]
    train_mean = train.mean()
    train_std = train.std()

    train = (train - train_mean) / train_std
    dev = (dev - train_mean) / train_std
    test = (test - train_mean) / train_std
    return train, dev, test

train, dev, test = split_standardize_jena(T)

# Windowing
import preprocessing as pre
windower = pre.WindowGenerator(train, dev, test)

# Build model
from sklearn.ensemble import IsolationForest

for element in windower.train.take(1):
    print(element[0].shape, element[1].shape)
    X, y = element

iForest = IsolationForest(random_state=420).fit(X)

for element in windower.dev.take(1):
    X_dev, y_dev = element

y_hat = iForest.predict(X_dev)
print(f"prediction: {y_hat}")
