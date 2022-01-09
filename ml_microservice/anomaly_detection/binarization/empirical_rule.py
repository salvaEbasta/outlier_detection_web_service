import pathlib
import joblib
from math import sqrt
import os
import re

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from ml_microservice import configuration as cfg

class EmpiricalRule(BaseEstimator, TransformerMixin):
    def __init__(self, k = 3):
        self.k = k

    def fit(self, X, y = None):
        """ 
            Params:
            - X: predicted values, from prediction task
            - y: real values, label of prediction
        """
        if y is None:
            self.mean_ = np.mean(X)
            self.var_ = np.var(X)
        else:
            e = y - X
            self.mean_ = np.mean(e)
            self.var_ = np.var(e)
        self.std_var_ = sqrt(self.var_)
        return self
    
    def transform(self, X):
        if not hasattr(self, "mean_") or \
            not hasattr(self, "var_") or \
            not hasattr(self, "std_var_"):
            raise RuntimeError("Need to be fitted first")

        return np.array(np.logical_or(
            np.less(X, self.mean_ - self.k * self.std_dev_), 
            np.greater(X, self.mean_ + self.k * self.std_dev_)
        ), dtype = int)
    
    def save(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        f_path = os.path.join(
            path_dir, 
            cfg.empRule.default_file
        )
        joblib.dump(self, f_path, compress = 3)
