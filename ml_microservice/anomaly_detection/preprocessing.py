import numpy as np
import tensorflow as tf
from tensorflow import keras

def split(dataframe, dev=True):
    """dataframe -> train, dev, test (porzione, no X o y)"""
    # TODO: le finestre che sto creando in test (e dev) mi fanno perdere degli istanti preziosi. Aggiorna la procedura
    n = len(dataframe)
    if dev:
        return dataframe[:int(n*.7)], dataframe[int(n*.7):int(n*.9)], dataframe[int(n*.9):]
    else:
        return dataframe[:int(n*.9)], dataframe[int(n*.9):]

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
        # Introduce noise
        return X, y
    
    @property
    def mean(self):
      return self._mean
    
    @property
    def std(self):
      return self._std

    @property
    def train(self):
        return self.make_dataset(self._train)
    
    @property
    def dev(self):
        if self._dev is None:
            return None
        return self.make_dataset(self._dev)
    
    @property
    def test(self):
        if self._test is None:
            return None
        return self.make_dataset(self._test)

    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            X, y = self.train
            result = (X[0], y[0])
            self._example = result
        return result
