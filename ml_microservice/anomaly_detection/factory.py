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
        self.factory = {
            # WindowedGaussian -----------------------------
            f"{WindowedGaussian.__name__}": {
                f"{cfg.factory['loader_k']}": loaders.WindGaussLoader,
                f"{cfg.factory['tuner_k']}": tuners.WindGaussTuner,
            },
            # DeepAnT --------------------------------------
            f"{DeepAnT.__name__}": {
                f"{cfg.factory['loader_k']}": loaders.DeepAnTLoader,
                f"{cfg.factory['tuner_k']}": tuners.DeepAnTTuner,
            },
            # GRU -----------------------------------------
            f"{GRU.__name__}": {
                f"{cfg.factory['loader_k']}": loaders.GRULoader,
                f"{cfg.factory['tuner_k']}": tuners.GRUTuner,
            },
            # LSTM ----------------------------------------
            f"{LSTM.__name__}": {
                f"{cfg.factory['loader_k']}": loaders.LSTMLoader,
                f"{cfg.factory['tuner_k']}": tuners.LSTMTuner,
            },
            # SARIMAX -------------------------------------
            f"{SARIMAX.__name__}": {
                f"{cfg.factory['loader_k']}": loaders.SARIMAXLoader,
                f"{cfg.factory['tuner_k']}": tuners.SARIMAXTuner,
            },
            # Prophet ---------------------------------------
            f"{Prophet.__name__}": {
                f"{cfg.factory['loader_k']}": loaders.ProphetLoader,
                f"{cfg.factory['tuner_k']}": tuners.ProphetTuner,
            },
        }
    