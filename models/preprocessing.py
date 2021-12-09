# %%
import tensorflow as tf
from tensorflow import keras
# %%
import os
os.chdir('..')
os.listdir()
# %%
import configparser
conf = configparser.ConfigParser()
conf.read('config.ini')

import pandas as pd

csv_file = 'jena_climate_2009_2016.csv'
dframe = pd.read_csv(os.path.join(conf['Resources']['datasetsPath'], csv_file))

# %%
dframe.shape
dframe = dframe[5::6]
# %%
temp_list = dframe['T (degC)'].squeeze()
# %%
n = len(temp_list)
train_set = temp_list[:int(n*.7)]
train_mean = train_set.mean()
train_std = train_set.std()

dev_set = temp_list[int(n*.7):int(n*.9)]
test_set = temp_list[int(n*.9):]
# %%
train_set = (train_set - train_mean) / train_std
dev_set = (dev_set - train_mean) / train_std
test_set = (test_set - train_mean) / train_std

# %%
import numpy as np

class WindowGenerator():
    def __init__(self, input_width=10, shift=1, label_width=1,
                train_set=train_set, dev_set=dev_set, test_set=test_set):
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
        self.train_set = train_set
        self.dev_set = dev_set
        self.test_set = test_set
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}'])

w1 = WindowGenerator(input_width=100)

def split_window(self, window):
    input_w = window[:, :-1]
    input_w.set_shape([None, self.input_width])
    
    label_w = window[:,-1]
    label_w.set_shape([None, self.label_width])
    return input_w, label_w

WindowGenerator.split_window = split_window

def make_dataset(self, data):
    data = np.array(data, dtype=float)
    dset = keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,
    )
    dset = dset.map(self.split_window)
    return dset

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
    return self.make_dataset(self.train_set)

@property
def dev(self):
    return self.make_dataset(self.dev_set)

@property
def test(self):
    return self.make_dataset(self.test_set)

WindowGenerator.train = train
WindowGenerator.dev = dev
WindowGenerator.test = test
# %%
for value in w1.train.take(1):
    print(f"{value}")
# %%
