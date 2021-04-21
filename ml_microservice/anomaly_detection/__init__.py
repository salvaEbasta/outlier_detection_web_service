import os
import configparser
import pandas as pd

def load_jena_temp():
    ##
    # return: jena['T (degC)'], jena
    # ##
    conf = configparser.ConfigParser()
    conf.read('config.ini')

    csv_file = os.path.join(conf['Resources']['datasetsPath'], 'jena_climate_2009_2016.csv')
    jena_df = pd.read_csv(csv_file)
    return jena_df['T (degC)'], jena_df

def split_standardize(dset):
    n = len(dset)
    train, dev, test = dset[:int(n*.7)], dset[int(n*.7):int(n*.9)], dset[int(n*.9):]
    train_mean = train.mean()
    train_std = train.std()

    train = (train - train_mean) / train_std
    dev = (dev - train_mean) / train_std
    test = (test - train_mean) / train_std
    return train, dev, test

def jena_full_pipeline():
    return split_standardize(load_jena_temp()[0])