import inspect
import joblib
import json
import os
import re

import numpy as np
from sklearn.base import TransformerMixin
from tensorflow.keras import models

from ml_microservice.anomaly_detection.models.deepant import DeepAnT

from . import configuration as cfg
from .detector import AnomalyDetector
from .models import windowed_gaussian as wg

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
        preload_file = cfg.deepant["preload_file"],
        params_file = cfg.deepant["params_file"],
        classifier_dir = cfg.deepant["classifier_dir"],
        forecast_dir = cfg.deepant["forecast_dir"]
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
        wg = joblib.load(os.path.join(path, self.file))
        return wg
