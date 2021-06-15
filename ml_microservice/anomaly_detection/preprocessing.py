import json
import os
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from ml_microservice import constants

def split(dataframe, dev=True, window_size=0):
    """
        dataframe -> train, dev, test (porzione, no X o y)\n
        if window_size is specified, dev and test will overlap the previous set.
    """
    if window_size < 0:
        raise ValueError('Window size has to be a value greater than 0')
    elif window_size > len(dataframe):
        raise ValueError(f'Window size ({window_size}) exceed dataframe length ({len(dataframe)})')
    
    n = len(dataframe)
    if dev:
        return dataframe[:int(n*.7)], \
            dataframe[int(n*.7) - window_size:int(n*.9)], \
            dataframe[int(n*.9) - window_size:]
    else:
        return dataframe[:int(n*.9)], \
            dataframe[int(n*.9) - window_size:]

class Preprocessor():
    """Class to preprocess datasets that contains a sequence.
        The set must be already partitioned
    """
    def __init__(self, train, dev=None, test=None, input_width=10, label_width=1):
        self.input_width = input_width
        self.label_width = label_width
        self.total_width = input_width + label_width

        self._train = train
        self._dev = dev
        self._test = test

        self._mean = np.mean(np.array(train))
        self._std = np.std(np.array(train))

    def extract_windows(self, data):
        """ D x 1 -> N x in_size x 1, N x label_size x 1 """
        data = np.array(data)
        if len(data) < self.total_width:
            tmp = np.zeros((self.total_width,))
            for i in range(len(data)):
                tmp[len(tmp) - len(data) + i] = data[i]
            data = tmp
        total_windows = len(data) - self.total_width + 1
        X = np.empty((total_windows, self.input_width))
        y = np.empty((total_windows, self.label_width))
        for i in range(total_windows):
            X[i] = data[i : i + self.input_width]
            y[i] = data[i + self.input_width : i + self.total_width]
        return X, y

    def standardize(self, data, mean, std):
        if std > 0:
            data = (data - mean) / std
        else:
            data = (data - mean)
        return data

    def make_dataset(self, data):
        """ D x 1 -> N x in_size x 1, N x label_size x 1 """
        data = np.array(data)
        # Standardization
        data = self.standardize(data, self._mean, self._std)
        # Make windows
        X, y = self.extract_windows(data)
        return X, y
    
    def augment_dataset(self, X, y):
        """
            Augment the dataset by duplicating instances and introducing gaussian noise.\n
            X.shape: [D, ...]  , y: [D, ...] -> [2*D, ...] , [2*D, ...]
        """
        X = np.vstack((X, 
                       X + np.random.normal(0, 0.1, size=X.shape),
                       X + np.indices(X.shape).sum(axis=0)%2*np.random.normal(0, 0.1, size=X.shape),
                       ))
        y = np.vstack((y, y, y))
        tmp = np.c_[X.reshape(len(X), -1), y.reshape(len(y), -1)]
        np.random.shuffle(tmp)
        return tmp[:, :X.size//len(X)].reshape(X.shape), \
            tmp[:, X.size//len(X):].reshape(y.shape)

    @property
    def mean(self):
      return self._mean
    
    @property
    def std(self):
      return self._std

    @property
    def train(self):
        self._last_train = self.make_dataset(self._train)
        return self._last_train
    
    @property
    def augmented_train(self):
        self._last_augmented_train = self.augment_dataset(*self.train)
        return self._last_augmented_train

    @property
    def dev(self):
        if self._dev is None:
            return None
        else:
            self._last_dev = self.make_dataset(self._dev)
            return self._last_dev
    
    @property
    def test(self):
        if self._test is None:
            return None
        else:
            self._last_test = self.make_dataset(self._test)
            return self._last_test
    
    @property
    def last_train(self):
        return getattr(self, '_last_train', None)

    @property
    def last_augmented_train(self):
        return getattr(self, '_last_agumented_train', None)

    @property
    def last_dev(self):
        return getattr(self, '_last_dev', None)
    
    @property
    def last_test(self):
        return getattr(self, '_last_test', None)

    @property
    def params(self):
        return dict(
            input_width = self.input_width,
            label_width = self.label_width,
            mean = self._mean,
            std = self._std,
        )

    def load_params(self, path):
        params = None
        if not os.path.exists(path):
            logging.warning("Path {:s} does not exists".format(path))
        else:
            with open(path, 'r') as f:
                params = json.load(f)
            self.input_width = params['input_width']
            self.label_width = params['label_width']
            self.total_width = self.input_width + self.label_width

            self._mean = params['mean']
            self._std = params['std']
    
    def save_params(self, ddir):
        if not os.path.exists(ddir):
            os.makedirs(ddir)
        f = os.path.join(ddir, constants.files.preprocessing_params)
        with open(f, 'w') as f:
            json.dump(self.params, f)


class SeriesFilter():
    def __init__(self, 
                    dframe,
                    min_datapoints = constants.seriesFilter.min_d_points, 
                    patience = constants.seriesFilter.patience, 
                ):
        """  """
        self._dframe = dframe
        self.min_datapoints = min_datapoints
        self._patience = patience

        self._dropped = pd.DataFrame()
        self._filtered = pd.DataFrame()
    
    def enough_datapoints(self, serie):
        """ serie: flat serie """
        return len(serie) >= self.min_datapoints
    
    def recent_information_content(self, serie):
        """ serie: flat serie """
        return any(serie[-self._patience:] > .0) or any(serie[-self._patience:] < .0)
    
    def filter_out(self, dframe):
        dropped = pd.DataFrame()
        dframe = pd.DataFrame(dframe)
        for i, col in enumerate(dframe.columns):
            if not self.recent_information_content(dframe[col]):
                dropped[col] = dframe[col]
                dframe = dframe.drop(col, axis=1)
        return dframe, dropped
    
    @property
    def dframe(self):
        self._filtered, self._dropped = self.filter_out(self._dframe)
        return self._filtered
    
    @property
    def dropped_cols(self):
        return self._dropped

class Padder():
    def __init__(self, serie, padding_length):
        self._serie = np.array(serie)
        self._pad_length = padding_length
    
    def zero_pre_loading(self):
        return self.pre_loading(np.zeros((self._pad_length,) + self._serie.shape[1:]))
    
    def pre_loading(self, as_padding):
        """ as_padding: np.array """
        as_padding = np.array(as_padding)
        if len(as_padding) < self._pad_length:
            return np.r_[
                np.zeros((self._pad_length - len(as_padding),) + self._serie.shape[1:]),
                as_padding,
                self._serie
            ]
        return np.r_[as_padding[-self._pad_length:], self._serie]
