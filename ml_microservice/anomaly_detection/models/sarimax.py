import inspect
import json
import os

import numpy as np
import pandas as pd
import pmdarima
from scipy import signal
from statsmodels import api as sm

from .. import configuration as cfg
from ..transformers import Preprocessor
from ..detector import AnomalyDetector, Forecaster
from .windowed_gaussian import WindowedGaussian

def get_exogenous(datetimeIDX, date_format = "%Y-%m-%d %H:%M:%S"):
    if date_format == "%Y-%m-%d %H:%M:%S":
        X = pd.get_dummies(
                pd.DataFrame(
                    {
                        'week_of_year': datetimeIDX.isocalendar().week,
                        #'hour_of_day': data.index.hour.astype('category'),
                        #'Intercept': np.ones_like(datetimeIDX),
                    }
                )
            ).astype('float64')
        #, index=datetimeIDX))
    X.index = range(len(datetimeIDX))
    return X

def get_seasonality(X, period = "W"):
    #fs, P = signal.periodogram(X)
    #f = fs[np.argmax(P)]
    #p = int(1 / (f + np.finfo(float).eps))
    #print(f"Period: {p}")
    #if p == X.shape[0]:
    #    return False, 1
    #return True, p
    if period == "W":
        return True, 52


class SARIMAX(AnomalyDetector, Forecaster):
    def __init__(
        self, gauss_win = 32, gauss_step = 16, order = [0,0,0],
        seasonal_order = [0,0,0,0],
    ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val) 

        self.forecaster = None
        self.classifier = None

    def fit(self, ts):
        #preprocessing
        pre = Preprocessor(ts)
        ts = pre.nan_filled
        #tmp = ts.set_index(cfg.cols["timestamp"])
        #tmp.index = pd.DatetimeIndex(tmp.index).to_period("W")
        #tmp = tmp[cfg.cols["X"]]

        endo = ts[cfg.cols["X"]]
        exo = get_exogenous(
            pd.DatetimeIndex(ts[cfg.cols["timestamp"]])
        )#.drop("Intercept", axis = 1, errors = 'ignore')
        print(exo)
        s, m = get_seasonality(endo.to_numpy())

        #order
        try:
            arima = pmdarima.auto_arima(
                y = endo, 
                exogenous = exo,
                max_p = 3,
                max_q = 3,
                error_action = 'ignore', 
                trace = True,
                suppress_warnings = True, 
                maxiter = 1,
                maxorder = 5,
                seasonal = s,
                m = m
            )
        except ValueError:
            arima = pmdarima.auto_arima(
                y = endo, 
                exogenous = exo,
                D = 0,
                max_p = 3,
                max_q = 3,
                error_action = 'ignore', 
                trace = True,
                suppress_warnings = True, 
                maxiter = 1,
                maxorder = 5,
                seasonal = s, 
                m = m
            )
        self.order = arima.order
        self.seasonal_order = arima.seasonal_order
        try:
            self.forecaster = sm.tsa.SARIMAX(
                endo,
                exo,
                order = self.order,
                seasonal_order = self.seasonal_order,
                time_varying = True, 
                mle_regression = False
            ).fit()
        except np.linalg.LinAlgError as err:
            self.forecaster = sm.tsa.SARIMAX(
                endo,
                exo,
                order = self.order,
                seasonal_order = self.seasonal_order,
                time_varying = True, 
                mle_regression = False,
                enforce_stationarity = False, 
                simple_differencing = True
            ).fit()

        y_hat = self.forecaster.predict().to_numpy()
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

    def forecast(self, ts):
        pre = Preprocessor(ts)
        ts = pre.nan_filled
        X = ts[cfg.cols["X"]].copy().to_numpy()
        exo = get_exogenous(
            pd.DatetimeIndex(ts[cfg.cols["timestamp"]])
        )
        y_hat = self.forecaster.forecast(steps = len(ts), exog = exo)

        residuals = pd.DataFrame()
        residuals[cfg.cols["X"]] = ts[cfg.cols["X"]]
        if cfg.cols["timestamp"] in ts.columns:
            residuals[cfg.cols["timestamp"]] = ts[cfg.cols["timestamp"]]
        residuals[cfg.cols["forecast"]] = y_hat
        residuals[cfg.cols["residual"]] = X - y_hat.to_numpy()
        return residuals
    
    def save(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        
        forecaster_path = os.path.join(path_dir, cfg.forecaster_model["forecaster_dir"])
        if not os.path.exists(forecaster_path):
            os.makedirs(forecaster_path)
        sm_file = os.path.join(forecaster_path, cfg.sarimax["statsmodels_file"])
        self.forecaster.save(sm_file)

        classifier_path = os.path.join(path_dir, cfg.forecaster_model["classifier_dir"])
        if not os.path.exists(forecaster_path):
            os.makedirs(forecaster_path)
        self.classifier.save(classifier_path)
        
        params_path = os.path.join(path_dir, cfg.forecaster_model["params_file"])
        with open(params_path, "w") as f:
            json.dump(self.get_params(), f)
        