import time

import pandas as pd

from ml_microservice import configuration as cfg
from ml_microservice.logic.detector_lib import Environment

from .loaders import Loader
from .tuners import Tuner
from .evaluators import Evaluator
from .preprocessing import Preprocessor
from .detector import AnomalyDetector

class Trainer():
    def __init__(self, env: Environment):
        self.env = env
        self.preproc = None

    def train(self, tuner: Tuner, ts: pd.DataFrame) -> AnomalyDetector:
        t0 = time.time()
        self.train_time_ = 0
        self.preproc = Preprocessor(ts)
        tuner.tune(self.preproc)

        self.last_train_IDX_ = self.preproc.last_train_IDX_
        if hasattr(self.preproc, "last_dev_IDX_"):
            self.last_dev_IDX_ = self.preproc.last_train_IDX_
        self.train_time_ = time.time() - t0
        return tuner.best_model_

    def predict(self, loader: Loader, ts: pd.DataFrame):
        t0 = time.time()

        model = loader.load(self.env.assets)
        self.preproc = Preprocessor(ts)
        self.preproc.for_prediction()
        y_hat = model.predict(self.preproc.values)

        if hasattr(model, "forecast_"):
            self.forecast_ = getattr(model, "forecast_")
        if hasattr(model, "predict_prob_"):
            self.predict_prob_ = getattr(model, "predict_prob_")
        self.prediction_time_ = time.time() - t0
        return y_hat

    def evaluate(self, evaluator: Evaluator, ts: pd.DataFrame, 
                    tsIDX: int, forget: bool = True):
        """ -> y_hat, f_hat"""
        t0 = time.time()
        evaluator.load_model(self.env.assets)

        self.preproc = Preprocessor(ts, start_from = tsIDX)
        evaluator.evaluate(self.preproc, forget)

        if hasattr(evaluator, "forecast_"):
            self.forecast_ = getattr(evaluator, "forecast_")
        if hasattr(evaluator, "predict_prob_"):
            self.predict_prob_ = getattr(evaluator, "predict_prob_")
        self.eval_time_ = time.time() - t0
        return evaluator.prediction_
    
    def ts_values(self):
        if self.preproc is None:
            return None
        return self.preproc.values
    
    def ts_dates(self):
        if self.preproc is None:
            return None
        return self.preproc.dates

class Supervisor:
    @staticmethod
    def train():
        pass

    @staticmethod
    def predict():
        pass

    @staticmethod
    def evaluate():
        pass

    @staticmethod
    def evaluate_train():
        pass

    @staticmethod
    def 