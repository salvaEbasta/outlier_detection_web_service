import math
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .. import configuration as cfg
from ..transformers import Preprocessor
from ..detector import AnomalyDetector

def q_function(x, mean, std_dev):
    """
    Q-function: https://www.gaussianwaves.com/2012/07/q-function-and-error-functions/
    """
    if x < mean:
        x = 2*mean - x
        return q_function(x, mean, std_dev)
    z = (x - mean) / (std_dev + np.finfo(float).eps)
    return 0.5 * math.erfc(z / math.sqrt(2))

class WindowedGaussian(AnomalyDetector):
    """
    Can be used to obtain normalized anomaly score from residuals
    Use: .fit, X = y - y_hat
    """
    def __init__(self, w = 32, step = 16):
        self.w = w
        self.step = step

        self.t = .95

    def fit(self, ts):
        p = Preprocessor(ts)
        ts = p.nan_filled

        self.window_ = []
        self.buffer_ = []
        self.mean_ = 0
        self.std_dev_ = 1
        self.errors_ = []
        
        scores = []
        for x in self._X(ts):
            score = .0
            self.errors_.append(x - self.mean_)
            if len(self.window_) > 0:
                score = 1 - q_function(x, self.mean_, self.std_dev_)
            scores.append(score)

            if len(self.window_) < self.w:
                self.window_.append(x)
                self.mean_ = np.mean(self.window_)
                self.std_dev_ = np.std(self.window_)
                #if self.std_dev_ == 0.0:
                #    self.std_dev_ = .000001
                continue
            self.buffer_.append(x)
            if len(self.buffer_) == self.step:
                self.window_ = self.window_[self.step:]
                self.window_.extend(self.buffer_)
                self.buffer_ = []
                self.mean_ = np.mean(self.window_)
                self.std_dev_ = np.std(self.window_)
                #if self.std_dev_ == 0.0:
                #    self.std_dev_ = .000001
        scores = np.array(scores)
        if cfg.cols["y"] in ts:
            y = ts[cfg.cols["y"]].to_numpy()
            fpr, tpr, thrs = roc_curve(y, scores)
            J = tpr - fpr
            idx = np.argmax(J)
            self.t = thrs[idx]
        return self

    def predict_proba(self, ts):
        if not hasattr(self, "window_") or \
            not hasattr(self, "buffer_") or \
            not hasattr(self, "mean_") or \
            not hasattr(self, "std_dev_"):
            raise RuntimeError("Must be fitted first")
        
        p = Preprocessor(ts)
        ts = p.nan_filled
        
        window = list(self.window_)
        buffer = list(self.buffer_)
        mean = self.mean_
        std_dev = self.std_dev_
        self.errors_ = []

        anomaly_scores = []
        for x in self._X(ts):
            score = .0
            self.errors_.append(x - mean)
            if len(window) > 0:
                score = 1 - q_function(x, mean, std_dev)
            anomaly_scores.append(score)

            if len(window) < self.w:
                window.append(x)
                mean = np.mean(window)
                std_dev = np.std(window)
                #if std_dev == 0.0:
                #    std_dev = .000001  
                continue
            buffer.append(x)
            if len(buffer) == self.step:
                window = window[self.step:]
                window.extend(buffer)
                buffer = []
                mean = np.mean(window)
                std_dev = np.std(window)
                #if std_dev == 0.0:
                #    std_dev = .000001
        res = pd.DataFrame()
        if cfg.cols["timestamp"] in ts.columns:
            res[cfg.cols["timestamp"]] = ts[cfg.cols["timestamp"]]
        res[cfg.cols["X"]] = ts[cfg.cols["X"]]
        res[cfg.cols["pred_prob"]] = np.array(anomaly_scores)
        return res
    
    def predict(self, ts):
        pp = self.predict_proba(ts)
        pp[cfg.cols["y"]] = np.array(
            np.greater(
                pp[cfg.cols["pred_prob"]].to_numpy(), 
                self.t
            ), 
            dtype = int
        )
        return pp

    def fit_predict(self, ts):
        self.window_ = []
        self.buffer_ = []
        self.mean_ = 0
        self.std_dev_ = 1
        return self.predict(ts)
    
    def save(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        path = os.path.join(path_dir, cfg.windGauss["default_file"])
        joblib.dump(self, path, compress = 3)
