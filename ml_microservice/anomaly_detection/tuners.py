import json
import os

from ml_microservice import configuration as cfg

from .preprocessing import Preprocessor

class Tuner():
    def tune(self, preprocessor: Preprocessor):
        raise NotImplementedError()
    
    def save_results(self, path_dir):
        raise NotImplementedError()

class AbstractTuner(Tuner):
    def __init__(self):
        self._search_space = {}
        self.best_model = None
        self.best_config = None
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
