import inspect
import joblib
import json
import os
import re

import numpy as np
from sklearn.base import TransformerMixin
from tensorflow.keras import models
import statsmodels.api as sm

from . import configuration as cfg
from .detector import AnomalyDetector
from .models.deepant import DeepAnT
from .models.gru import GRU
from .models.lstm import LSTM
from .models.sarimax import SARIMAX
from .models.prophet import Prophet


class Loader():
    def load(self, path) -> AnomalyDetector:
        raise NotImplementedError()

class EmpRuleLoader(Loader):
    def __init__(self, file = cfg.empRule["default_file"]) -> None:
        self.file = file

    def load(self, path):
        if re.match(cfg.empRule["file_ext"].format(".+"), self.file) is None:
            return None
        if self.file not in os.listdir(path):
            return None
        return joblib.load(os.path.join(path, self.file))

class WindGaussLoader(Loader):
    def __init__(self, file = cfg.windGauss["default_file"]):
        self.file = file
        
    def load(self, path):
        if re.match(cfg.windGauss["file_ext"].format(".+"), self.file) is None:
            return None
        if self.file not in os.listdir(path):
            return None
        return joblib.load(os.path.join(path, self.file))

class DeepAnTLoader(Loader):
    def __init__(self, 
        preload_file = cfg.forecaster_model["preload_file"],
        params_file = cfg.forecaster_model["params_file"],
        classifier_dir = cfg.forecaster_model["classifier_dir"],
        forecaster_dir = cfg.forecaster_model["forecast_dir"]
    ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
    
    def load(self, path) -> AnomalyDetector:
        if self.preload_file not in os.listdir(path) or \
            self.params_file not in os.listdir(path) or \
                self.forecaster_dir not in os.listdir(path) or \
                    self.classifier_dir not in os.listdir(path):
            return None
        d = DeepAnT()
        params_path = os.path.join(path, self.params_file)
        with open(params_path, "r") as f:
            params = json.load(params_path)
        d.set_params(**params)

        preload_path = os.path.join(path, self.preload_file)
        with open(preload_path, "r") as f:
            preload = np.array(json.load(f))
        setattr(d, "preload", preload)

        classifier_path = os.path.join(path, self.classifier_dir)
        erLoader = EmpRuleLoader()
        setattr(d, "classifier", erLoader.load(classifier_path))
        
        forecaster_path = os.path.join(path, self.forecaster_dir)
        setattr(d, "forecaster", models.load_model(forecaster_path))
        return d

class GRULoader(Loader):
    def __init__(self, 
        preload_file = cfg.forecaster_model["preload_file"],
        params_file = cfg.forecaster_model["params_file"],
        classifier_dir = cfg.forecaster_model["classifier_dir"],
        forecaster_dir = cfg.forecaster_model["forecast_dir"]
    ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
    
    def load(self, path) -> AnomalyDetector:
        if self.preload_file not in os.listdir(path) or \
            self.params_file not in os.listdir(path) or \
                self.forecaster_dir not in os.listdir(path) or \
                    self.classifier_dir not in os.listdir(path):
            return None
        gru = GRU()
        params_path = os.path.join(path, self.params_file)
        with open(params_path, "r") as f:
            params = json.load(params_path)
        gru.set_params(**params)

        preload_path = os.path.join(path, self.preload_file)
        with open(preload_path, "r") as f:
            preload = np.array(json.load(f))
        setattr(gru, "preload", preload)

        classifier_path = os.path.join(path, self.classifier_dir)
        erLoader = EmpRuleLoader()
        setattr(gru, "classifier", erLoader.load(classifier_path))
        
        forecaster_path = os.path.join(path, self.forecaster_dir)
        setattr(gru, "forecaster", models.load_model(forecaster_path))
        return gru

class LSTMLoader(Loader):
    def __init__(self, 
        preload_file = cfg.forecaster_model["preload_file"],
        params_file = cfg.forecaster_model["params_file"],
        classifier_dir = cfg.forecaster_model["classifier_dir"],
        forecaster_dir = cfg.forecaster_model["forecast_dir"]
    ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
    
    def load(self, path) -> AnomalyDetector:
        if self.preload_file not in os.listdir(path) or \
            self.params_file not in os.listdir(path) or \
                self.forecaster_dir not in os.listdir(path) or \
                    self.classifier_dir not in os.listdir(path):
            return None
        lstm = LSTM()
        params_path = os.path.join(path, self.params_file)
        with open(params_path, "r") as f:
            params = json.load(params_path)
        lstm.set_params(**params)

        preload_path = os.path.join(path, self.preload_file)
        with open(preload_path, "r") as f:
            preload = np.array(json.load(f))
        setattr(lstm, "preload", preload)

        classifier_path = os.path.join(path, self.classifier_dir)
        erLoader = EmpRuleLoader()
        setattr(lstm, "classifier", erLoader.load(classifier_path))
        
        forecaster_path = os.path.join(path, self.forecaster_dir)
        setattr(lstm, "forecaster", models.load_model(forecaster_path))
        return lstm

class SARIMAXLoader(Loader):
    def __init__(self, 
        params_file = cfg.forecaster_model["params_file"],
        classifier_dir = cfg.forecaster_model["classifier_dir"],
        forecaster_dir = cfg.forecaster_model["forecast_dir"],
        statsmodels_file = cfg.sarimax["statsmodels_file"]
    ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
    
    def load(self, path) -> AnomalyDetector:
        if self.params_file not in os.listdir(path) or \
            self.forecaster_dir not in os.listdir(path) or \
                self.statsmodels_file not in os.listdir(
                    os.path.join(path, self.forecaster_dir)
                ) or self.classifier_dir not in os.listdir(path):
            return None
        sarimax = SARIMAX()
        params_path = os.path.join(path, self.params_file)
        with open(params_path, "r") as f:
            params = json.load(params_path)
        sarimax.set_params(**params)

        classifier_path = os.path.join(path, self.classifier_dir)
        wgLoader = WindGaussLoader()
        classifier = wgLoader.load(classifier_path)
        setattr(sarimax, "classifier", classifier)
        
        forecaster_path = os.path.join(path, self.forecaster_dir)
        forecaster_file = os.path.join(forecaster_path, self.statsmodels_file)
        forecaster = sm.load(forecaster_file)
        setattr(sarimax, "forecaster", forecaster)
        return sarimax

class ProphetLoader(Loader):
    def __init__(self, 
        preload_file = cfg.forecaster_model["preload_file"],
        params_file = cfg.forecaster_model["params_file"],
        classifier_dir = cfg.forecaster_model["classifier_dir"],
        forecaster_dir = cfg.forecaster_model["forecast_dir"],
        forecaster_file = cfg.prophet["file"]
    ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
    
    def load(self, path) -> AnomalyDetector:
        if self.params_file not in os.listdir(path) or \
            self.forecaster_dir not in os.listdir(path) or \
                self.forecaster_file not in os.listdir(
                    os.path.join(path, self.forecaster_dir)
                ) or self.classifier_dir not in os.listdir(path):
            return None
        prophet = Prophet()
        params_path = os.path.join(path, self.params_file)
        with open(params_path, "r") as f:
            params = json.load(params_path)
        prophet.set_params(**params)

        classifier_path = os.path.join(path, self.classifier_dir)
        wgLoader = WindGaussLoader()
        classifier = wgLoader.load(classifier_path)
        setattr(prophet, "classifier", classifier)
        
        forecaster_path = os.path.join(path, self.forecaster_dir)
        forecaster_file = os.path.join(forecaster_path, self.statsmodels_file)
        forecaster = joblib.load(forecaster_file)
        setattr(prophet, "forecaster", forecaster)
        return prophet
