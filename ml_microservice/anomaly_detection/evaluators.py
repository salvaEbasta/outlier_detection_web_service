import os

import pandas as pd

from ml_microservice import configuration as cfg
from ml_microservice.logic.detector_lib import Environment
from .detector import AnomalyDetector
from .preprocessing import Preprocessor

def load_history(path_dir) -> pd.DataFrame:
    if not os.path.exists(path_dir):
        h = pd.DataFrame()
        h[cfg.evaluator.date_column] = []
        return pd.DataFrame()
    
    h_path = os.path.join(path_dir, cfg.evaluator.history_file)
    return pd.read_csv(h_path)

class Evaluator:
    def evaluate(self, preprocessor: Preprocessor, forget = True):
        """
        -> {score: float, }
        """
        raise NotImplementedError()

    def is_performance_dropping(self):
        raise NotImplementedError()

    def load_model(self, path_dir):
        raise NotImplementedError()

    def save_results(self, path_dir):
        raise NotImplementedError()

class AbstractEvaluator(Evaluator):
    def __init__(self, env: Environment):
        super().__init__()
        self.env = env
        self.history = load_history(env.root)

    def save_history(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        h_path = os.path.join(path_dir, cfg.evaluator.history_file)
        self.history.to_csv(h_path, index = False)
    
    @property
    def model_(self) -> AnomalyDetector:
        return getattr(self, "_model", None)

class WindGaussEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()
    
    def evaluate(self, preprocessor: Preprocessor, forget=True):
        #if preprocessor.
        return {}