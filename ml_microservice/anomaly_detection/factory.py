from . import configuration as cfg
from . import loaders
from . import tuners
from .models import windowed_gaussian as wg


class Factory():
    def has(self, type: str):
        return type in self.factory

    def get_types(self):
        return [k for k in self.factory.keys()]

    def get_tuner(self, type: str) -> tuners.Tuner:
        if type not in self.factory:
            return None
        return self.factory[type][cfg.factory["tuner_k"]]()

    def get_loader(self, type: str) -> loaders.Loader:
        if type not in self.factory:
            return None
        return self.factory[type][cfg.factory["loader_k"]]()

    def __init__(self):
        self.factory = {}
        
        # WindowedGaussian -----------------------------
        winGauss = wg.WindowedGaussian.__name__
        self.factory[winGauss][cfg.factory["loader_k"]] = loaders.WindGaussLoader
        self.factory[winGauss][cfg.factory["tuner_k"]] = tuners.WindGaussTuner
