from ml_microservice import configuration as cfg
from . import loaders
from . import tuners
from .models import windowed_gaussian as wg


class Factory():
    def get_types(self):
        return [k for k in self.factory.keys()]

    def get_tuner(self, type: str):
        if type not in self.factory:
            return None
        return self.factory[type][cfg.factory.tuner_k]

    def get_loader(self, type: str):
        if type not in self.factory:
            return None
        return self.factory[type][cfg.factory.loader_k]

    def __init__(self):
        self.factory = {}
        self.factory[wg.WindowedGaussian.__name__] \
            [cfg.factory.loader_k] = loaders.WindGaussLoader
        self.factory[wg.WindowedGaussian.__name__] \
            [cfg.factory.tuner_k] = tuners