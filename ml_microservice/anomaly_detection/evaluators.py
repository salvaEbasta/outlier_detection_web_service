from math import nan
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error

from ml_microservice.logic.detector_lib import Environment
from .metrics import naive_metric, naive_prediction

from . import configuration as cfg

def load_history(path_dir) -> pd.DataFrame:
    if path_dir is None or not os.path.exists(path_dir):
        h = pd.DataFrame()
        h[cfg.cols["timestamp"]] = ""
        h["f1"] = ""
        h["precision"] = ""
        h["recall"] = ""
        h["mse"] = ""
        h["rmse"] = ""
        h["naive"] = ""
        return pd.DataFrame()
    
    h_path = os.path.join(path_dir, cfg.evaluator["history_file"])
    h = pd.read_csv(h_path)
    h[cfg.cols["timestamp"]] = pd.to_datetime(h[cfg.cols["timestamp"]])
    return 

class Evaluator:
    def evaluate(self, model, ts):
        """
        -> {score: float, }
        """
        raise NotImplementedError()

    def is_performance_dropping(self):
        raise NotImplementedError()

    def save_results(self, path_dir):
        raise NotImplementedError()

class GPEvaluator(Evaluator):
    """
    History frame:
    --------------
    Timestamp - MSE - RMSE - NAIVE - F1 - PRECISION - RECALL
    """
    def __init__(self, env: Environment = None):
        self.env = env
        self.history = load_history(env.root if env is not None else None)

    def save_history(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        h_path = os.path.join(path_dir, cfg.evaluator["history_file"])
        self.history.to_csv(h_path, index = False)
    
    def evaluate(self, model, ts):
        f1 = np.nan
        prec = np.nan
        recall = np.nan
        mse = np.nan
        rmse = np.nan
        naive = np.nan

        if cfg.cols["y"] in ts.columns:
            X_ts = ts.drop(cfg.cols["y"], axis = 1)
        else:
            X_ts = ts
        self.prediction_  = model.predict(X_ts)
        y_hat = self.prediction_[cfg.cols["y"]].to_numpy()
        
        if cfg.cols["y"] in ts.columns:
            y = ts[cfg.cols["y"]].to_numpy()
            f1 = f1_score(y, y_hat)
            prec = precision_score(y, y_hat)
            recall = recall_score(y, y_hat)

        if cfg.cols["forecast"] in self.prediction_.columns:
            forecast_ = self.prediction_[cfg.cols["forecast"]].to_numpy()
            X = X_ts[cfg.cols["X"]].to_numpy()
            mse = mean_squared_error(X, forecast_)
            rmse = mean_squared_error(X, forecast_, squared = False)
            naive = naive_metric(X, forecast_, naive_prediction(X))
        self.scores_ = {
            "f1": f1,
            "precision": prec,
            "recall": recall,
            "mse": mse,
            "rmse": rmse,
            "naive": naive
        }
        return self.scores_
    
    def keep_last_scores(self):
        if not hasattr(self, "scores_"):
            return
        self.scores_[cfg.cols["timestamp"]] = pd.to_datetime("today").normalize()
        self.history = self.history.append(self.scores_, ignore_index = True)

    def is_performance_dropping(self, win = 3):
        """
        naive score
        """
        naive_drop = False
        if self.history["naive"].isnull().sum() == 0:
            naive_drop = self.detect_degradation(
                self.history["naive"].to_numpy(), win
            )
        
        f1_drop = False
        if self.history["f1"].isnull().sum() != 0:
            f1_drop = self.detect_degradation(
                -self.history["f1"].to_numpy(), win
            )
        return naive_drop or f1_drop

    def detect_degradation(self, x, w = 3):
        global_mean = np.mean(x)
        grad = np.gradient(x)
        win_mean = global_mean if x.shape[0] < w \
                    else np.mean(x[-w:])
        return global_mean < win_mean and np.mean(grad) > 0
