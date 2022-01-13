from . import configuration as cfg
from . import loaders
from . import tuners
from .models.windowed_gaussian import WindowedGaussian
from .models.deepant import DeepAnT
from .models.gru import GRU
from .models.lstm import LSTM
from .models.sarimax import SARIMAX
from .models.prophet import Prophet


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
        winGauss = WindowedGaussian.__name__
        self.factory[winGauss][cfg.factory["loader_k"]] = loaders.WindGaussLoader
        self.factory[winGauss][cfg.factory["tuner_k"]] = tuners.WindGaussTuner

        # DeepAnT --------------------------------------
        dant = DeepAnT.__name__
        self.factory[dant][cfg.factory["loader_k"]] = loaders.DeepAnTLoader
        self.factory[dant][cfg.factory["tuner_k"]] = tuners.DeepAnTTuner

        # GRU -----------------------------------------
        gru = GRU.__name__
        self.factory[gru][cfg.factory["loader_k"]] = loaders.GRULoader
        self.factory[gru][cfg.factory["tuner_k"]] = tuners.GRUTuner

        # LSTM ----------------------------------------
        lstm = LSTM.__name__
        self.factory[lstm][cfg.factory["loader_k"]] = loaders.LSTMLoader
        self.factory[lstm][cfg.factory["tuner_k"]] = tuners.LSTMTuner

        # SARIMAX -------------------------------------
        sarimax = SARIMAX.__name__
        self.factory[sarimax][cfg.factory["loader_k"]] = loaders.SARIMAXLoader
        self.factory[sarimax][cfg.factory["tuner_k"]] = tuners.SARIMAXTuner

        # Prophet
        prophet = Prophet.__name__
        self.factory[prophet][cfg.factory["loader_k"]] = loaders.ProphetLoader
        self.factory[prophet][cfg.factory["tuner_k"]] = tuners.ProphetTuner