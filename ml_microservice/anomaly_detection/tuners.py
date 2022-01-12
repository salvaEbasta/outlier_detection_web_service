import itertools
import json
import os

from sklearn.metrics import f1_score
from tensorflow.keras import callbacks
from kerastuner import Hyperband

from . import configuration as cfg
from .transformers import Preprocessor
from .models.windowed_gaussian import WindowedGaussian
from .models import deepant

class Tuner():
    def tune(self, ts):
        """
        Must set:
        ---------
        - self.explored_cfgs_ = dict
        - self.best_config_ = dict
        - self.best_model_ = AnomalyDetector
        - self.best_score_ = float

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
    
    def set_space(self, configs):
        for k, v in configs:
            if k in self._search_space:
                self._search_space[k] = v

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
        self.explored_cfgs_ = []

        if cfg.cols["y"] not in ts:
            X = ts.drop(cfg.cols["y"], axis = 1)
            self.best_model_ = WindowedGaussian().fit(X)
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
        X = ts.drop(cfg.cols["y"], axis = 1)
        
        for config in configs:
            current = WindowedGaussian()
            current.set_params(config)
            current.fit(X)
            y_hat = current.predict(X)
            score = f1_score(y, y_hat[cfg.cols["y"]].to_numpy())
            
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


class DeepAnTTuner(AbstractTuner):
    def __init__(self):
        super().__init__(search_space = dict(
            win = [16, 32, 64], 
            maps = [32, 64], 
            kernel = [2, 3, 4], 
            conv_strides = [1, 2, 4], 
            pool_kernel = [2, 4], 
            conv_layers = [2, 4, 8, 16], 
            hidden_size = [128, 256, 512],
            dropout_rate = [.2, .4],
            gauss_win = [16, 32, 64, 128],
            gauss_step = [8, 16, 32, 64]
        ))
    
    def tune(self, ts):
        self.explored_cfgs_ = []
        pre = Preprocessor(ts)
        ts = pre.nan_filled
        # tune forecaster
        X, y = pre.extract_windows(
            ts[cfg.cols["X"]].copy().to_numpy(), 
            w = self.win
        )
        X_train, X_dev = pre.train_test_split(X)
        y_train, y_dev = pre.train_test_split(y)
        X_train, y_train = pre.shuffle(X_train, y_train)
        X_dev, y_dev = pre.shuffle(X_dev, y_dev)

        def hypermodel(hp):
            return deepant.DeepAnT_forecaster(
                win = hp.Choice("win", values = self.search_space["win"]),
                maps = hp.Choice("maps", values = self.search_space["maps"]),
                kernel = hp.Choice("kernel", values = self.search_space["kernel"]),
                conv_strides = hp.Choice("conv_strides", values = self.search_space["conv_strides"]),
                pool_kernel = hp.Choice("pool_kernel", values = self.search_space["pool_kernel"]),
                conv_layers = hp.Choice("conv_layers", values = self.search_space["conv_layers"]),
                hidden_size = hp.Choice("hidden_size", values = self.search_space["hidden_size"]),
                dropout_rate = hp.Float(
                    "dropout_rate", 
                    min_value = self.search_space["dropout_rate"][0],
                    max_value = self.search_space["dropout_rate"][1],
                ),
            )

        hb = Hyperband(hypermodel, factor = 5, seed = 42)
        hb.search(
            X_train, y_train,
            validation_data = (X_dev, y_dev),
            max_epochs = 50,
            callbacks = [
                callbacks.EarlyStopping(
                    monitor = 'val_loss',
                    patience = 5,
                ), ]
        )
        best_hps = hb.get_best_hyperparameters()[0]
        self.best_config_ = {k: v  for k, v in best_hps.values.items()}
        self.win = self.best_config_["win"]

        forecaster = hb.hypermodel.build(best_hps)
        history = forecaster.fit(
            X_train, y_train, 
            batch_size = 2,
            epochs = 50,
            validation = (X_dev, y_dev),
            callbacks = [
                callbacks.EarlyStopping(
                    monitor = 'val_loss',
                    patience = 5,
                ), ]
        )
        
        val_loss_history = history.history['val_loss']
        best_epoch_ = val_loss_history.index(min(val_loss_history)) + 1
        self.forecaster = hb.hypermodel.build(best_hps)
        X, y = pre.extract_windows(
            ts[cfg.cols["X"]].copy().to_numpy(), 
            w = self.win
        )
        X, y = pre.shuffle(X, y)
        self.forecaster.fit(X, y, epochs = best_epoch_)

        wgTuner = WindGaussTuner()
        wgTuner.set_space({
            "w": self.search_space["gauss_win"],
            "step": self.search_space["gauss_step"],
        })
        wgTuner.tune(ts)
        self.classifier = wgTuner.best_model_
        for k, v in wgTuner.best_config_.items():
            self.best_config_[k] = v
        
        self.explored_cfgs_.append({
            "predictor": self.best_config_,
            "classifier": wgTuner.explored_cfgs_,
        })

        self.preload = ts[cfg.cols["X"]].copy().to_numpy()[-self.win : ]

        model = deepant.DeepAnT()
        model.set_params(self.best_config_)
        model.forecaster = self.forecaster
        model.classifier = self.classifier
        model.preload = self.preload
        self.best_model_ = model
        self.best_score_ = 0
        return self
