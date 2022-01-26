import inspect
import json
import os

import numpy as np
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import callbacks

from .. import configuration as cfg
from ..transformers import Preprocessor
from ..detector import AnomalyDetector, Forecaster
from .windowed_gaussian import WindowedGaussian
from ..metrics import naive_prediction

def lstm_forecaster(win = 32, size1 = 128, dropout1 = .45, rec_dropout1 = .45,
                    size2 = 128, dropout2 = .45, rec_dropout2 = .45, 
                    forecasting_horizon = 1) -> Model:
        i = layers.Input(shape = [win, ])
        i = layers.GaussianNoise(1) (i)
        r = layers.Reshape([win, 1], input_shape = [win,]) (i)
        lstm = layers.LSTM(
            units = size1,
            return_sequences = True,
            input_shape = [None, 1],
            dropout = dropout1,
            recurrent_dropout = rec_dropout1,
        ) (r)
        lstm = layers.LSTM(
            units = size2,
            dropout = dropout2,
            recurrent_dropout = rec_dropout2,
        ) (lstm)
        f = layers.Flatten() (lstm)
        o = layers.Dense(units = forecasting_horizon) (f)
        m = Model(inputs = i, outputs = o, name = "LSTM")
        m.compile(
            optimizer = optimizers.Adam(),
            loss = losses.MeanSquaredError()
        )
        return m

class LSTM(AnomalyDetector, Forecaster):
    def __init__(self, gauss_win = 32, gauss_step = 16, win = 32, 
                    size1 = 128, dropout1 = .45, rec_dropout1 = .45,
                    size2 = 128, dropout2 = .45, rec_dropout2 = .45
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
        X, y = pre.extract_windows(
            ts[cfg.cols["X"]].copy().to_numpy(), 
            w = self.win
        )
        X_train, X_dev = pre.train_test_split(X)
        y_train, y_dev = pre.train_test_split(y)
        X_train, y_train = pre.shuffle(X_train, y_train)
        X_dev, y_dev = pre.shuffle(X_dev, y_dev)

        self.forecaster = lstm_forecaster(
            win = self.win, size1 = self.size1, dropout1 = self.dropout1, 
            rec_dropout1 = self.rec_dropout1, size2 = self.size2, 
            dropout2 = self.dropout2, rec_dropout2 = self.rec_dropout2
        )
        self.forecaster.fit(
            X_train, y_train, 
            batch_size = 2,
            epochs = 50,
            validation = (X_dev, y_dev),
            callbacks = [
                callbacks.EarlyStopping(
                    monitor = 'val_loss',
                    patience = 5,
                ), ]
        )

        X, y = pre.extract_windows(
            ts[cfg.cols["X"]].copy().to_numpy(), 
            w = self.win
        )
        y_hat = np.empty([len(ts), ])
        y_hat[ : self.win] = naive_prediction(ts[cfg.cols["X"]].copy().to_numpy())
        y_hat[self.win : ] = self.forecaster.predict(X).flatten()
        
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

        self.preload = ts[cfg.cols["X"]].copy().to_numpy()[-self.win : ]
        return self
    
    def predict_proba(self, ts):
        residuals = self.forecast(ts)
        residuals[cfg.cols["X"]] = residuals[cfg.cols["residuals"]]
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
        X = np.empty([self.preload.shape[0] + len(ts), ])
        X[ : self.win] = self.preload
        X[self.win : ] = ts[cfg.cols["X"]].copy().to_numpy()
        wins, y = pre.extract_windows(X, w = self.win)
        
        y_hat = self.forecaster.predict(wins).flatten()

        residuals = pd.DataFrame()
        residuals[cfg.cols["X"]] = ts[cfg.cols["X"]]
        if cfg.cols["timestamp"] in ts.columns:
            residuals[cfg.cols["timestamp"]] = ts[cfg.cols["timestamp"]]
        residuals[cfg.cols["forecast"]] = y_hat
        residuals[cfg.cols["residual"]] = y.flatten() - y_hat
        return residuals
    
    def save(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        
        forecaster_path = os.path.join(path_dir, cfg.forecaster_model["forecaster_dir"])
        if not os.path.exists(forecaster_path):
            os.makedirs(forecaster_path)
        self.forecaster.save(forecaster_path)

        classifier_path = os.path.join(path_dir, cfg.forecaster_model["classifier_dir"])
        if not os.path.exists(forecaster_path):
            os.makedirs(forecaster_path)
        self.classifier.save(classifier_path)

        preload_path = os.path.join(path_dir, cfg.forecaster_model["preload_file"])
        with open(preload_path, "w") as f:
            json.dump(self.preload.tolist(), f)
        
        params_path = os.path.join(path_dir, cfg.forecaster_model["params_file"])
        with open(params_path, "w") as f:
            json.dump(self.get_params(), f)
        