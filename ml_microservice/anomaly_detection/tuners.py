import itertools
import json
import os

from sklearn.metrics import f1_score

from . import configuration as cfg
from .transformers import Preprocessor
from .models.windowed_gaussian import WindowedGaussian

class Tuner():
    def tune(self, ts):
        """
        Must set:
        ---------
        self.explored_cfgs_ = dict
        self.best_config_ = dict
        self.best_model_ = AnomalyDetector
        self.best_score_ = float

        Returns:
        --------
         -> self
        """
        raise NotImplementedError()
    
    def save_results(self, path_dir):
        raise NotImplementedError()

class AbstractTuner(Tuner):
    def __init__(self, search_space):
        self._search_space = search_space
        self.explored_cfgs_ = []

    @property
    def search_space(self):
        return self._search_space

    def to_explore(self):
        keys, values = zip(*self._search_space.items())
        configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return configs

    def save_results(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        explored_path = os.path.join(path_dir, cfg.tuner["results_file"])
        with open(explored_path, "w") as f:
            json.dump(self.explored_cfgs, f)


class WindGaussTuner(AbstractTuner):
    def __init__(self):
        super().__init__(search_space = {
            "w": [32, 64, 128, 256],
            "k": [16, 32, 64, 128],
        })

    def tune(self, ts):
        if cfg.cols["y"] not in ts:
            self.best_model_ = WindowedGaussian()
            self.best_config_ = self.best_model_.get_params()
            self.best_score_ = 0
            return self
        #Gridsearch
        self.best_model_ = None
        self.best_config_ = self.best_model_.get_params()
        self.best_score_ = 0

        self.explored_cfgs_ = []
        configs = self.to_explore()
        y = ts[cfg.cols["y"]].to_numpy()
        X = ts.drop(cfg.cols[""], axis = 1)
        for config in configs:
            current = WindowedGaussian()
            current.set_params(config)
            y_hat = current.predict(X)
            score = f1_score(y, y_hat)
            
            if self.best_score_ < score:
                self.best_score_ = score
                self.best_model_ = current
                self.best_config_ = config
            
            exploration_result = {
                "config": config,
                "f1": score
            }
            self.explored_cfgs_.append(exploration_result)
        return self
