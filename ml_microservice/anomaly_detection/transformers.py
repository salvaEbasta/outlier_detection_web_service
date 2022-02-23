import numpy as np
import pandas as pd

from ml_microservice import configuration as old_cfg
from . import configuration as cfg

class Transformer:
    def fit(self, ts):
        raise NotImplementedError()
    
    def transform(self, ts):
        raise NotImplementedError()

class Preprocessor():
    def __init__(self, 
        ts: pd.DataFrame,
        start_from = 0,
        value_col = cfg.cols["X"],
        date_col = cfg.cols["timestamp"],
        nan_ratio = cfg.preprocessor["good_nan_ratio"]
    ):
        self._valC = value_col
        self._dateC = date_col
        self.nan_ratio = nan_ratio

        self.startIDX = start_from
        self.ts = ts[self.startIDX:].copy(deep = True)
    
    def fill_nan(self, ts):
        nan_num = ts[self._valC].isnull().sum()
        if nan_num / len(ts) <= self.nan_ratio:
            ts[self._valC] = ts[self._valC].fillna(method = "ffill").fillna(method = "bfill")
        else:
            ts[self._valC] = ts[self._valC].fillna(0.0)
        return ts
    
    @property
    def nan_filled(self):
        return self.fill_nan(self.ts)

    def train_test_split(self, data, train_ratio = 0.7):
        return data[ : int(train_ratio * len(data))], data[int(train_ratio * len(data)) : ]

    @property
    def train_test(self):
        return self.train_test_split(self.ts)

    def shuffle(self, X, y):
        tmp = np.c_[X.reshape(len(X), -1), y.reshape(len(y), -1)]
        np.random.shuffle(tmp)
        return tmp[:, :X.size//len(X)].reshape(X.shape), \
            tmp[:, X.size//len(X):].reshape(y.shape)

    def extract_windows(self, data, w, fh = 1):
        total_width = w + fh
        if len(data) < total_width:
            tmp = np.zeros([total_width, ])
            for i in range(len(data)):
                tmp[len(tmp) - len(data) + i] = data[i]
            data = tmp
        total_windows = len(data) - total_width + 1
        X = np.empty([total_windows, w])
        y = np.empty([total_windows, fh])
        for i in range(total_windows):
            X[i] = data[i : i + w]
            y[i] = data[i + w : i + total_width]
        return X, y
    
    def _augment_dataset(self, X, y):
        """
            Augment the dataset by duplicating instances and introducing gaussian noise.\n
            X.shape: [D, ...]  , y: [D, ...] -> [2*D, ...] , [2*D, ...]
        """
        X = np.vstack((
            X, 
            X + np.random.normal(0, 0.1, size = X.shape),
            X + np.indices(X.shape).sum(axis = 0) % 2 * np.random.normal(0, 0.1, size = X.shape),
        ))
        y = np.vstack((y, y, y))
        tmp = np.c_[X.reshape(len(X), -1), y.reshape(len(y), -1)]
        np.random.shuffle(tmp)
        return tmp[:, :X.size//len(X)].reshape(X.shape), \
            tmp[:, X.size//len(X):].reshape(y.shape)
