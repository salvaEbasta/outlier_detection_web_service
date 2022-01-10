import math
import joblib
import os

import numpy as np

from ml_microservice import configuration as cfg
from ..detector import AnomalyDetector

def q_function(mean, std_dev, x):
    """
    Q-function: https://www.gaussianwaves.com/2012/07/q-function-and-error-functions/
    """
    if x < mean:
        x = 2*mean - x
        return q_function(mean, std_dev, x)
    z = (x - mean) / std_dev
    return 0.5 * math.erfc(z / math.sqrt(2))

class WindowedGaussian(AnomalyDetector):
    """
    Can be used to obtain normalized anomaly score from residuals
    Use: .fit, X = y - y_hat
    """
    def __init__(self, w = 32, step = 16):
        self.w = w
        self.step = step

    def fit(self, X, y = None):
        """
        Params:
        - X: series
        - y: predictions, not used
        """
        self.window_ = []
        self.buffer_ = []
        self.mean_ = 0
        self.std_dev_ = 1
        self.errors_ = []

        for x in X:
            self.errors_.append(x - self.mean_)
            if len(self.window_) < self.w:
                self.window_.append(x)
                self.mean_ = np.mean(self.window_)
                self.std_dev_ = np.std(self.window_)
                if self.std_dev_ == 0.0:
                    self.std_dev_ = np.finfo(float).eps
                continue
            self.buffer_.append(x)
            if len(self.buffer_) == self.step:
                self.window_ = self.window_[self.step:]
                self.window_.extend(self.buffer_)
                self.buffer_ = []
                self.mean_ = np.mean(self.window_)
                self.std_dev_ = np.std(self.window_)
                if self.std_dev_ == 0.0:
                    self.std_dev_ = np.finfo(float).eps
        return self

    def predict_proba(self, X):
        if not hasattr(self, "window_") or \
            not hasattr(self, "buffer_") or \
            not hasattr(self, "mean_") or \
            not hasattr(self, "std_dev_"):
            raise RuntimeError("Must be fitted first")
        
        window = list(self.window_)
        buffer = list(self.buffer_)
        mean = self.mean_
        std_dev = self.std_dev_
        self.errors_ = []

        anomaly_scores = []
        for x in X:
            score = .0
            self.errors_.append(x - mean)
            if len(window) > 0:
                score = 1 - q_function(x, mean, std_dev)
            if len(window) < self.w:
                window.append(x)
                mean = np.mean(window)
                std_dev = np.std(window)
                if std_dev == 0.0:
                    std_dev = np.finfo(float).eps
                anomaly_scores.append(score)
                continue
            buffer.append(x)
            if len(buffer) == self.step:
                window = window[self.step:]
                window.extend(buffer)
                buffer = []
                mean = np.mean(window)
                std_dev = np.std(window)
                if std_dev == 0.0:
                    std_dev = np.finfo(float).eps
            anomaly_scores.append(score)
        return np.array(anomaly_scores)
    
    def predict(self, X):
        return np.array(
            np.greater(self.predict_proba(X), 0.5), 
            dtype = int
        )
    
    def fit_predict(self, X, y = None):
        self.window_ = []
        self.buffer_ = []
        self.mean_ = 0
        self.std_dev_ = 1
        return self.predict(X)
    
    def save(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        path = os.path.join(path_dir, cfg.windGauss.default_file)
        joblib.dump(self, path, compress = 3)
