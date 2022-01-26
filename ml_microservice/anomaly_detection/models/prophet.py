import inspect
import joblib
import json
import os

import numpy as np
import pandas as pd
import prophet

from .. import configuration as cfg
from ..transformers import Preprocessor
from ..detector import AnomalyDetector, Forecaster
from .windowed_gaussian import WindowedGaussian

def make_future_dataframe(periods, start_date, freq='D'):
    dates = pd.date_range(
        start = start_date,
        periods = periods + 1,
        freq = freq)
    #dates = dates[dates > start_date]
    dates = dates[:periods]
    return pd.DataFrame({'ds': dates})

class Prophet(AnomalyDetector, Forecaster):
    def __init__(self, gauss_win = 32, gauss_step = 16, 
                    changepoint_prior_scale = 0.05,
                    seasonality_prior_scale = 10,
                    ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val) 

        self.preload = None
        self.forecaster = None
        self.classifier = None
    
    def fit(self, ts):
        #preprocessing
        pre = Preprocessor(ts)
        ts = pre.nan_filled
        #tmp = ts.set_index(cfg.cols["timestamp"])
        #tmp.index = pd.DatetimeIndex(tmp.index).to_period("W")
        #tmp = tmp[cfg.cols["X"]]

        prophet_ts = pd.DataFrame()
        prophet_ts[cfg.prophet["timestamp"]] = ts[cfg.cols["timestamp"]]
        prophet_ts[cfg.prophet["value"]] = ts[cfg.cols["X"]]
        self.forecaster = prophet.Prophet(
            changepoint_prior_scale = self.changepoint_prior_scale,
            seasonality_prior_scale = self.seasonality_prior_scale
        )
        self.forecaster.fit(prophet_ts)
        #future = self.forecaster.make_future_dataframe(periods = 0, freq = "W")
        #future = make_future_dataframe(
        #    periods = len(ts), 
        #    start_date = ts[cfg.cols["timestamp"]].min(), 
        #    freq = "W"
        #)
        future = pd.DataFrame({"ds": ts[cfg.cols["timestamp"]]})
        prophet_res = self.forecaster.predict(future)

        y_hat = prophet_res["yhat"].to_numpy()
        #print(f"y_hat: {y_hat}")
        
        residuals = pd.DataFrame()
        residuals[cfg.cols["timestamp"]] = ts[cfg.cols["timestamp"]]
        residuals[cfg.cols["X"]] = ts[cfg.cols["X"]].copy().to_numpy() - y_hat
        if cfg.cols["y"] in ts.columns:
            residuals[cfg.cols["y"]] = ts[cfg.cols["y"]]
        
        self.classifier = WindowedGaussian(
            self.gauss_win,
            self.gauss_step
        )
        self.classifier.fit(residuals)
        return self
    
    def predict_proba(self, ts):
        residuals = self.forecast(ts)
        residuals[cfg.cols["X"]] = residuals[cfg.cols["residual"]]
        predict_proba = self.classifier.predict_proba(residuals)
        
        predict_proba[cfg.cols["X"]] = ts[cfg.cols["X"]]
        if cfg.cols["timestamp"] in ts.columns:
            if cfg.cols["timestamp"] not in predict_proba.columns:
                predict_proba[cfg.cols["timestamp"]] = ts[cfg.cols["timestamp"]]
        predict_proba[cfg.cols["forecast"]] = residuals[cfg.cols["forecast"]]
        return predict_proba
    
    def forecast(self, ts):
        pre = Preprocessor(ts)
        ts = pre.nan_filled
        X = ts[cfg.cols["X"]].copy().to_numpy()
        
        future = self.forecaster.make_future_dataframe(periods = len(ts), freq = "W")
        prophet_res = self.forecaster.predict(future)
        y_hat = prophet_res["yhat"].to_numpy()

        residuals = pd.DataFrame()
        residuals[cfg.cols["X"]] = ts[cfg.cols["X"]]
        if cfg.cols["timestamp"] in ts.columns:
            residuals[cfg.cols["timestamp"]] = ts[cfg.cols["timestamp"]]
        residuals[cfg.cols["forecast"]] = y_hat
        residuals[cfg.cols["residual"]] = X - y_hat
        return residuals
    
    def predict(self, ts):
        residuals = self.forecast(ts)
        residuals[cfg.cols["X"]] = residuals[cfg.cols["residual"]]
        prediction = self.classifier.predict(residuals)
        
        prediction[cfg.cols["X"]] = ts[cfg.cols["X"]]
        if cfg.cols["timestamp"] in ts.columns:
            if cfg.cols["timestamp"] not in prediction.columns:
                prediction[cfg.cols["timestamp"]] = ts[cfg.cols["timestamp"]]
        prediction[cfg.cols["forecast"]] = residuals[cfg.cols["forecast"]]
        return prediction

    def save(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        
        forecaster_path = os.path.join(path_dir, cfg.forecaster_model["forecaster_dir"])
        if not os.path.exists(forecaster_path):
            os.makedirs(forecaster_path)
        prophet_file = os.path.join(forecaster_path, cfg.prophet["file"])
        joblib.dump(self.forecaster, prophet_file, compress = 3)

        classifier_path = os.path.join(path_dir, cfg.forecaster_model["classifier_dir"])
        if not os.path.exists(forecaster_path):
            os.makedirs(forecaster_path)
        self.classifier.save(classifier_path)
        
        params_path = os.path.join(path_dir, cfg.forecaster_model["params_file"])
        with open(params_path, "w") as f:
            json.dump(self.get_params(), f)
        