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

def DeepAnT_forecaster(win = 32, maps = 32, kernel = 2, conv_strides = 1, 
                        pool_kernel = 2, conv_layers = 2, hidden_size = 256,
                        dropout_rate = 0.4, forecasting_horizon = 1) -> Model:
        i = layers.Input(shape = [win, ])
        i = layers.GaussianNoise(1) (i)
        r = layers.Reshape([win, 1], input_shape = [win,]) (i)
        pool = r
        for _ in range(conv_layers):
            conv = layers.Conv1D(
                filters = maps, 
                kernel_size = kernel,
                strides = conv_strides, 
                activation = 'relu',
                padding = "same"
            ) (pool)
            pool = layers.MaxPool1D(pool_size = pool_kernel, padding = "valid") (conv)
        f = layers.Flatten() (pool)
        d = layers.Dense(units = hidden_size, activation = 'relu', kernel_initializer="he_normal") (f)
        d = layers.Dropout(dropout_rate) (d)
        o = layers.Dense(units = forecasting_horizon) (d)
        m = Model(inputs = i, outputs = o, name = "DeepAnT")
        m.compile(
            optimizer = optimizers.Adam(),
            loss = losses.MeanSquaredError()
        )
        return m

class DeepAnT(AnomalyDetector, Forecaster):
    def __init__(self, gauss_win = 32, gauss_step = 16, win = 32, maps = 32,
                    kernel = 2, conv_strides = 1, pool_kernel = 2, conv_layers = 2,
                    hidden_size = 256, dropout_rate = .4
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

        self.forecaster = DeepAnT_forecaster(
            win = self.win, maps = self.maps, kernel = self.kernel, 
            conv_strides = self.conv_strides, pool_kernel = self.pool_kernel,
            conv_layers = self.conv_layers, hidden_size = self.hidden_size,
            dropout_rate = self.dropout_rate
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
        