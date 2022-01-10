import json
import os

from ml_microservice import configuration as cfg
from ml_microservice.anomaly_detection.models.windowed_gaussian import WindowedGaussian

from .preprocessing import Preprocessor

class Tuner():
    def tune(self, preprocessor: Preprocessor):
        """
        Must update:
        self.explored_cfgs = dict
        Must set:
        self.best_config_ = dict
        self.best_model_ = AnomalyDetector
        self.best_score_ = float
        Return:
         -> self
        """
        raise NotImplementedError()
    
    def save_results(self, path_dir):
        raise NotImplementedError()

class AbstractTuner(Tuner):
    def __init__(self):
        self._search_space = {}
        self.explored_cfgs = []

    @property
    def search_space(self):
        return self._search_space

    def save_results(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        explored_path = os.path.join(path_dir, cfg.tuner.results_file)
        with open(explored_path, "w") as f:
            json.dump(self.explored_cfgs, f)

class WindGaussTuner(AbstractTuner):
    def __init__(self):
        super().__init__()
        self._search_space = {
            "w": [32, 64, 128, 256],
            "k": [16, 32, 64, 128],
        }

    def tune(self, preprocessor: Preprocessor):
        self.best_model_ = WindowedGaussian()
        self.best_config_ = self.best_model_.get_params()
        self.best_score_ = 0
        return self