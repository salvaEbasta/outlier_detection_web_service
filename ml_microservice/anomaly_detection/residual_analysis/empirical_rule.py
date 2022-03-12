import joblib
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ..detector import Persistent

from .. import configuration as cfg

class EmpiricalRule(BaseEstimator, Persistent):
    def __init__(self, k = 3, robust = False):
        self.k = k
        self.robust = robust

    def fit(self, ts):
        """ 
            Params:
            - ts: pandas.DataFrame
        """
        X = ts[cfg.cols["X"]].to_numpy()
        self.mean_ = np.mean(X) if not self.robust else np.median(X)
        self.var_ = np.var(X)
        self.std_dev_ = float(np.std(X))
        return self
    
    def predict(self, ts):
        if not hasattr(self, "mean_") or \
            not hasattr(self, "var_") or \
            not hasattr(self, "std_dev_"):
            raise RuntimeError("Need to be fitted first")

        X = ts[cfg.cols["X"]].to_numpy()
        ts[cfg.cols["y"]] = np.array(
            np.logical_or(
                np.less(X, self.mean_ - self.k * self.std_dev_), 
                np.greater(X, self.mean_ + self.k * self.std_dev_)
            ), 
            dtype = int
        )
        return ts
    
    def save(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        f_path = os.path.join(
            path_dir, 
            cfg.empRule["default_file"]
        )
        joblib.dump(self, f_path, compress = 3)
