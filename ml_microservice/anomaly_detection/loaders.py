import joblib
import os
import re

from sklearn.base import TransformerMixin

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
