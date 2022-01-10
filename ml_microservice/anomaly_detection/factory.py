from ml_microservice import configuration as cfg
from ml_microservice.anomaly_detection import evaluators
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
        return self.factory[type][cfg.factory.tuner_k]()

    def get_loader(self, type: str) -> loaders.Loader:
        if type not in self.factory:
            return None
        return self.factory[type][cfg.factory.loader_k]()
    
    def get_evaluator(self, type: str) -> evaluators.Evaluator:
        if type not in self.factory:
            return None
        return self.factory[type][cfg.factory.eval_k]()

    def __init__(self):
        self.factory = {}
        
        self.factory[wg.WindowedGaussian.__name__] \
            [cfg.factory.loader_k] = loaders.WindGaussLoader
        self.factory[wg.WindowedGaussian.__name__] \
            [cfg.factory.tuner_k] = tuners.WindGaussTuner
        self.factory[wg.WindowedGaussian.__name__] \
            [cfg.factory.eval_k] = evaluators.WindGaussEvaluator
        
        