import numpy as np
import tensorflow as tf
from tensorflow import keras

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
        X = np.zeros((total_windows, self.input_width))
        y = np.zeros((total_windows, self.label_width))
        for i in range(total_windows):
            X[i] = data[i : i + self.input_width]
            y[i] = data[i + self.input_width : i + self.total_width]
        return X, y

    def make_dataset(self, data):
        """ D x 1 -> N x in_size x 1, N x label_size x 1 """
        data = np.array(data)
        # Standardization
        if self._std > 0:
            data = (data - self._mean) / self._std
        else:
            data = (data - self._mean)
        # Make windows
        X, y = self.extract_windows(data)
        return X, y
    
    def augment_dataset(self, X, y):
        """
            Augment the dataset by duplicating instances and introducing gaussian noise.\n
            X.shape: [D, ...]  , y: [D, ...] -> [2*D, ...] , [2*D, ...]
        """
        X_augmented = np.copy(X)
        X_augmented = X_augmented + np.random.normal(0, 0.1, size=X_augmented.shape)
        X = np.concatenate((X, X_augmented))
        y = np.concatenate((y, np.copy(y)))
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
        self._last_train = self.augment_dataset(*self.train)
        return self._last_train

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
    def last_dev(self):
        return getattr(self, '_last_dev', None)
    
    @property
    def last_test(self):
        return getattr(self, '_last_test', None)