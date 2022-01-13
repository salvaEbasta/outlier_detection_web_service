import inspect
import joblib
import json
import os
import re

import numpy as np
from sklearn.base import TransformerMixin
from tensorflow.keras import models

from . import configuration as cfg
from .detector import AnomalyDetector
#from .models import windowed_gaussian
from .models.deepant import DeepAnT
from .models.gru import GRU
from .models.lstm import LSTM
from .models.sarimax import SARIMAX
from .models.prophet import Prophet


class Loader():
    def load(self, path) -> AnomalyDetector:
        raise NotImplementedError()

class WindGaussLoader(Loader):
    def __init__(self, file = cfg.windGauss["default_file"]):
        self.file = file
        
    def load(self, path):
        if re.match(cfg.windGauss["file_ext"].format(".+"), self.file) is None:
            return None
        if self.file not in os.listdir(path):
            return None
        wg = joblib.load(os.path.join(path, self.file))
        return wg

class DeepAnTLoader(Loader):
    def __init__(self, 
        preload_file = cfg.forecaster_model["preload_file"],
        params_file = cfg.forecaster_model["params_file"],
        classifier_dir = cfg.forecaster_model["classifier_dir"],
        forecast_dir = cfg.forecaster_model["forecast_dir"]
    ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
    
    def load(self, path) -> AnomalyDetector:
        if self.preload_file not in os.listdir(path) or \
            self.params_file not in os.listdir(path) or \
                self.forecast_file not in os.listdir(path) or \
                    self.classifier_file not in os.listdir(path):
            return None
        d = DeepAnT()
        params_path = os.path.join(path, self.params_file)
        with open(params_path, "r") as f:
            params = json.load(params_path)
        d.set_params(params)

        preload_path = os.path.join(path, self.preload_file)
        with open(preload_path, "r") as f:
            preload = np.array(json.load(f))
        setattr(d, "preload", preload)

        classifier_path = os.path.join(path, self.classifier_dir)
        wgLoader = WindGaussLoader()
        classifier = wgLoader.load(classifier_path)
        setattr(d, "classifier", classifier)
        
        forecaster_path = os.path.join(path, self.forecaster_dir)
        forecaster = models.load_model(forecaster_path)
        setattr(d, "forecaster", forecaster)
        return d

class GRULoader(Loader):
    def __init__(self, 
        preload_file = cfg.forecaster_model["preload_file"],
        params_file = cfg.forecaster_model["params_file"],
        classifier_dir = cfg.forecaster_model["classifier_dir"],
        forecast_dir = cfg.forecaster_model["forecast_dir"]
    ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
    
    def load(self, path) -> AnomalyDetector:
        if self.preload_file not in os.listdir(path) or \
            self.params_file not in os.listdir(path) or \
                self.forecast_file not in os.listdir(path) or \
                    self.classifier_file not in os.listdir(path):
            return None
        gru = GRU()
        params_path = os.path.join(path, self.params_file)
        with open(params_path, "r") as f:
            params = json.load(params_path)
        gru.set_params(params)

        preload_path = os.path.join(path, self.preload_file)
        with open(preload_path, "r") as f:
            preload = np.array(json.load(f))
        setattr(gru, "preload", preload)

        classifier_path = os.path.join(path, self.classifier_dir)
        wgLoader = WindGaussLoader()
        classifier = wgLoader.load(classifier_path)
        setattr(gru, "classifier", classifier)
        
        forecaster_path = os.path.join(path, self.forecaster_dir)
        forecaster = models.load_model(forecaster_path)
        setattr(gru, "forecaster", forecaster)
        return gru

class LSTMLoader(Loader):
    def __init__(self, 
        preload_file = cfg.forecaster_model["preload_file"],
        params_file = cfg.forecaster_model["params_file"],
        classifier_dir = cfg.forecaster_model["classifier_dir"],
        forecast_dir = cfg.forecaster_model["forecast_dir"]
    ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
    
    def load(self, path) -> AnomalyDetector:
        if self.preload_file not in os.listdir(path) or \
            self.params_file not in os.listdir(path) or \
                self.forecast_file not in os.listdir(path) or \
                    self.classifier_file not in os.listdir(path):
            return None
        lstm = LSTM()
        params_path = os.path.join(path, self.params_file)
        with open(params_path, "r") as f:
            params = json.load(params_path)
        lstm.set_params(params)

        preload_path = os.path.join(path, self.preload_file)
        with open(preload_path, "r") as f:
            preload = np.array(json.load(f))
        setattr(lstm, "preload", preload)

        classifier_path = os.path.join(path, self.classifier_dir)
        wgLoader = WindGaussLoader()
        classifier = wgLoader.load(classifier_path)
        setattr(lstm, "classifier", classifier)
        
        forecaster_path = os.path.join(path, self.forecaster_dir)
        forecaster = models.load_model(forecaster_path)
        setattr(lstm, "forecaster", forecaster)
        return lstm

class SARIMAXLoader(Loader):
    def __init__(self, 
        preload_file = cfg.forecaster_model["preload_file"],
        params_file = cfg.forecaster_model["params_file"],
        classifier_dir = cfg.forecaster_model["classifier_dir"],
        forecast_dir = cfg.forecaster_model["forecast_dir"]
    ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
    
    def load(self, path) -> AnomalyDetector:
        return
        if self.preload_file not in os.listdir(path) or \
            self.params_file not in os.listdir(path) or \
                self.forecast_file not in os.listdir(path) or \
                    self.classifier_file not in os.listdir(path):
            return None
        lstm = LSTM()
        params_path = os.path.join(path, self.params_file)
        with open(params_path, "r") as f:
            params = json.load(params_path)
        lstm.set_params(params)

        preload_path = os.path.join(path, self.preload_file)
        with open(preload_path, "r") as f:
            preload = np.array(json.load(f))
        setattr(lstm, "preload", preload)

        classifier_path = os.path.join(path, self.classifier_dir)
        wgLoader = WindGaussLoader()
        classifier = wgLoader.load(classifier_path)
        setattr(lstm, "classifier", classifier)
        
        forecaster_path = os.path.join(path, self.forecaster_dir)
        forecaster = models.load_model(forecaster_path)
        setattr(lstm, "forecaster", forecaster)
        return lstm

class ProphetLoader(Loader):
    def __init__(self, 
        preload_file = cfg.forecaster_model["preload_file"],
        params_file = cfg.forecaster_model["params_file"],
        classifier_dir = cfg.forecaster_model["classifier_dir"],
        forecast_dir = cfg.forecaster_model["forecast_dir"]
    ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
    
    def load(self, path) -> AnomalyDetector:
        return
        if self.preload_file not in os.listdir(path) or \
            self.params_file not in os.listdir(path) or \
                self.forecast_file not in os.listdir(path) or \
                    self.classifier_file not in os.listdir(path):
            return None
        lstm = LSTM()
        params_path = os.path.join(path, self.params_file)
        with open(params_path, "r") as f:
            params = json.load(params_path)
        lstm.set_params(params)

        preload_path = os.path.join(path, self.preload_file)
        with open(preload_path, "r") as f:
            preload = np.array(json.load(f))
        setattr(lstm, "preload", preload)

        classifier_path = os.path.join(path, self.classifier_dir)
        wgLoader = WindGaussLoader()
        classifier = wgLoader.load(classifier_path)
        setattr(lstm, "classifier", classifier)
        
        forecaster_path = os.path.join(path, self.forecaster_dir)
        forecaster = models.load_model(forecaster_path)
        setattr(lstm, "forecaster", forecaster)
        return lstm

class TransformerLoader():
    def load(self, path) -> TransformerMixin:
        raise NotImplementedError()

class EmpRuleLoader(TransformerLoader):
    def __init__(self, file = cfg.empRule["default_file"]):
        self.file = file
        
    def load(self, path):
        if re.match(cfg.empRule["file_ext"].format(".+"), self.file) is None:
            return None
        if self.file not in os.listdir(path):
            return None
        er = joblib.load(os.path.join(path, self.file))
        return er
