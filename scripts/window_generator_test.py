import os
import configparser
conf = configparser.ConfigParser()
conf.read('config.ini')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

class WindowGenerator():
    def __init__(self, train=None, dev=None, test=None, input_width=10, shift=1, label_width=1):
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self._train = train
        self._dev = dev
        self._test = test
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}'])

    @tf.autograph.experimental.do_not_convert
    def split_window(self, window):
        x = window[:, self.input_slice]
        label = window[:, self.labels_slice]
        
        x.set_shape([None, self.input_width])
        label.set_shape([None, self.label_width])
        return x, label

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
    
    @property
    def train(self):
        if self._train is not None:
            return self.make_dataset(self._train)
        return self._train
    
    @property
    def dev(self):
        if self._dev is not None:
            return self.make_dataset(self._dev)
        return self._dev

    @property
    def test(self):
        if self._test is not None:
            return self.make_dataset(self._test)
        return self._test

    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result


if __name__ == "__main__":
    csv_file = os.path.join(conf['Datasets']['path'], 'jena_climate_2009_2016.csv')
    jena_df = pd.read_csv(csv_file)
    T = jena_df['T (degC)']
    T = T[5::6]

    w = WindowGenerator(train=T)
    #dset = w.make_dataset(T)
    print(f"{w.example[0].shape}")
    print(f"{w.example[1].shape}")