import os
import json
from typing import Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ml_microservice import strings
from ml_microservice.anomaly_detection import model_factory

class Detector():
    def __init__(self, window_size=50, l=0.01, k=3, regressor_model="linear", path=None):
        """ 
            Specify path to load a Detector from a folder. The other parameters will be ignored even if specified.\n
            l: lambda\n
            k: multiplicator for variance to compose the threshold\n
            window_size: the dimension of the input\n
            Assume the label to be monodimensional\n
        """
        if path is None:
            self._window_size = window_size
            
            self._regressor_model = regressor_model
            self._reg_builder = model_factory.RegressorFactory(window_size)
            if not self._reg_builder.has(regressor_model):
                raise ValueError(f'The regressor model selected, \'{regressor_model}\', is not supported')
            self._regressor = self._reg_builder.build(regressor_model)
            self._lambda = l
            self._k = k
            self._var = 0
        else:
            self._load(path)
    
    @property
    def window_size(self):
        return self._window_size

    @property
    def params(self) -> Dict:
        """
            A dict with all the parameters
        """
        return dict(
            window_size=self.window_size,
            regressor=self._regressor_model,
            l=self._lambda,
            k=self._k,
            variance=self._var,
        )

    def fit(self, X, y, validation_data: Tuple =None, epochs=1):
        """
            Sequences in X and y follow the temporal order
        """
        # train the regressor
        if validation_data is not None:
            history = self._regressor.fit(
                X, y, 
                validation_data=validation_data,
                epochs=epochs,
            )
        else:
            history = self._regressor.fit(X, y, epochs=epochs)

        # train the threshold
        self.update_variance(X, y)
        return history

    def update_variance(self, X, y):
        assert len(y.shape) == 2 and y.shape[1] == 1

        y_hat = self._regressor.predict(X)
        #y_hat = tf.expand_dims(y_hat, axis=-1)
        #if len(y_hat.shape) >2:
        #    y_hat = tf.squeeze(y_hat, axis=-1) 
        assert len(y_hat.shape) == 2 and y_hat.shape[1] == 1
        
        mse = tf.keras.losses.mean_squared_error(y_hat, y)
        mse = mse.numpy()
        self._var = mse[0]
        for i in range(1, mse.shape[0]):
            self._var = (1 - self._lambda) * self._var + self._lambda * mse[1]

    @property
    def threshold(self):
        return self._k * np.sqrt(self._var)

    def detect(self, X, y):
        """-> np.array(N x 1), 0: expected, 1: anomaly"""
        assert len(y.shape) == 2 and y.shape[1] == 1

        y_hat = self._regressor.predict(X)
        assert len(y_hat.shape) == 2 and y_hat.shape[1] == 1
        anomalies = np.abs(y_hat - y) > self.threshold
        return anomalies.astype(int)

    def detect_update(self, X, y):
        """
            Sequences in X and y respect the overall temporal order in which they are extracted
            1: anomaly, 0: expected
        """
        assert len(y.shape) == 2 and y.shape[1] == 1
        
        y_hat = self.predict(X)
        assert len(y_hat.shape) == 2 and y_hat.shape[1] == 1
        anomalies = []
        for i in range(y_hat.shape[0]):
            anomalies.append([np.abs(y_hat[i] - y[i]) > self.threshold])
            self._var = (1 - self._lambda) * self._var + self._lambda * (y_hat[i] - y[i])**2

        return np.array(anomalies).astype(int)

    def predict(self, X):
        return self._regressor.predict(X)

    def save(self, path):
        self._regressor.save(path)
        params = dict(
            l=self._lambda,
            window_size=self.window_size,
            k=self._k,
            variance=self._var,
            regressor_model=self._regressor_model,
        )
        with open(os.path.join(path, strings.detector_param_file), 'w') as f:
            json.dump(params, f)

    def _load(self, path):
        with open(os.path.join(path, strings.detector_param_file), 'r') as f:
            params = json.load(f)
        self._window_size = int(params['window_size'])
        self._lambda = float(params['l'])
        self._k = float(params['k'])
        self._var = np.float(params['variance'])

        self._regressor_model = params['regressor_model']
        self._reg_builder = model_factory.RegressorFactory(self.window_size)
        self._regressor = keras.models.load_model(path)