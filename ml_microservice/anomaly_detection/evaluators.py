from math import nan
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error

from ml_microservice import configuration as old_cfg
from ml_microservice.logic.detector_lib import Environment

from . import configuration as cfg
from .detector import AnomalyDetector
from .loaders import WindGaussLoader
from .transformers import Preprocessor

def load_history(path_dir) -> pd.DataFrame:
    if not os.path.exists(path_dir):
        h = pd.DataFrame()
        h[old_cfg.evaluator.date_column] = []
        return pd.DataFrame()
    
    h_path = os.path.join(path_dir, old_cfg.evaluator.history_file)
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
    Timestamp - MSE - RMSE - F1 - PRECISION - RECALL
    """
    def __init__(self, env: Environment):
        self.env = env
        self.history = load_history(env.root)

    def save_history(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        h_path = os.path.join(path_dir, old_cfg.evaluator.history_file)
        self.history.to_csv(h_path, index = False)
    
    def evaluate(self, model, ts):
        y = ts[cfg.cols["y"]].to_numpy()
        X = ts.drop(cfg.cols["y"], axis = 1)
        self.prediction_  = model.predict(X)
        f1 = f1_score(y, self.prediction_)
        prec = precision_score(y, self.prediction_)
        recall = recall_score(y, self.prediction_)

        if hasattr(model, "predict_prob_"):
            self.predict_prob_ = model.predict_prob_

        mse = np.nan
        rmse = np.nan
        if hasattr(model, "forecast_"):
            self.forecast_ = model.forecast_
            y = X[cfg.cols["X"]].to_numpy()
            mse = mean_squared_error(y, model.forecast_)
            rmse = mean_squared_error(y, model.forecast_, squared = False)
        self.scores_ = {
            "f1": f1,
            "precision": prec,
            "recall": recall,
            "mse": mse,
            "rmse": rmse,
        }
        return self.scores_
    
    def keep_last_scores(self):
        if not hasattr(self, "scores_"):
            return
        self.scores_[cfg.cols["timestamp"]] = pd.to_datetime("today").normalize()
        self.history = self.history.append(self.scores_, ignore_index = True)

    def is_performance_dropping(self):
        """
        naive score
        """
        pass