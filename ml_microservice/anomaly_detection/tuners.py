import itertools
import json
import os

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from tensorflow.keras import callbacks
from kerastuner import Hyperband, Objective

from . import configuration as cfg
from .transformers import Preprocessor
from .models.windowed_gaussian import WindowedGaussian
from .models import deepant, gru, lstm, sarimax, prophet

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
        for k, v in configs.items():
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
            "step": [16, 32, 64, 128],
        })

    def tune(self, ts):
        if cfg.cols["y"] not in ts:
            X = ts
            self.best_model_ = WindowedGaussian().fit(X)
            self.best_config_ = self.best_model_.get_params()
            self.best_score_ = -1
            self.explored_cfgs_ = [{
                "config": self.best_config_,
                "f1": -1
            },]
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
            current.set_params(**config)
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
            win = [26, ], 
            maps = [32, 64, ], 
            kernel = [2, 4, ], 
            conv_strides = [1, 2, ], 
            pool_kernel = [2, ], 
            conv_layers = [2, ], 
            hidden_size = [128, 256, ],
            dropout_rate = [.2, .4],
            gauss_win = [16, 32, 64, 128],
            gauss_step = [8, 16, 32, 64]
        ))
    
    def tune(self, ts):
        self.explored_cfgs_ = {}
        pre = Preprocessor(ts)
        ts = pre.nan_filled
        # tune forecaster
        X, y = pre.extract_windows(
            ts[cfg.cols["X"]].copy().to_numpy(), 
            w = self.search_space["win"][0]
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
                    step = 0.5,
                ),
            )

        hb = Hyperband(
            hypermodel, factor = 3, seed = 42, overwrite = True,
            objective = Objective("val_loss", direction="min"),
        )
        hb.search(
            X_train, y_train,
            validation_data = (X_dev, y_dev),
            epochs = 20,
            callbacks = [
                callbacks.EarlyStopping(
                    monitor = 'val_loss',
                    patience = 3,
                ), ]
        )
        best_hps = hb.get_best_hyperparameters()[0]
        self.best_config_ = {k: v  for k, v in best_hps.values.items()}
        self.win = self.best_config_["win"]

        forecaster = hb.hypermodel.build(best_hps)
        history = forecaster.fit(
            X_train, y_train, 
            batch_size = 2,
            epochs = 20,
            validation_data = (X_dev, y_dev),
            callbacks = [
                callbacks.EarlyStopping(
                    monitor = 'val_loss',
                    patience = 3,
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
        y_hat = self.forecaster.predict(X_dev).flatten()
        rmse = mean_squared_error(y_dev, y_hat, squared = False)
        forecaster_configs = [{
            "config": self.best_config_,
            "rmse": rmse,
        },]

        wgTuner = WindGaussTuner()
        wgTuner.set_space({
            "w": self.search_space["gauss_win"],
            "step": self.search_space["gauss_step"],
        })
        wgTuner.tune(ts)
        self.classifier = wgTuner.best_model_

        self.explored_cfgs_["predictor"] = forecaster_configs
        self.explored_cfgs_["classifier"] = wgTuner.explored_cfgs_

        self.preload = ts[cfg.cols["X"]].copy().to_numpy()[-self.win : ]

        model = deepant.DeepAnT()
        self.best_config_["gauss_win"] = wgTuner.best_config_["w"]
        self.best_config_["gauss_step"] = wgTuner.best_config_["step"]
        model.set_params(**self.best_config_)
        model.forecaster = self.forecaster
        model.classifier = self.classifier
        model.preload = self.preload
        self.best_model_ = model
        self.best_score_ = 0
        return self

class GRUTuner(AbstractTuner):
    def __init__(self):
        super().__init__(search_space = dict(
            win = [26, ], 
            size1 = [64, 128, 256],
            dropout1 = [.3, .5],
            rec_dropout1 = [.3, .5],
            size2 = [64, 128, 256],
            dropout2 = [.3, .5],
            rec_dropout2 = [.3, .5],
            gauss_win = [16, 32, 64, 128],
            gauss_step = [8, 16, 32, 64]
        ))
    
    def tune(self, ts):
        self.explored_cfgs_ = {}
        pre = Preprocessor(ts)
        ts = pre.nan_filled
        # tune forecaster
        X, y = pre.extract_windows(
            ts[cfg.cols["X"]].copy().to_numpy(), 
            w = self.search_space["win"][0]
        )
        X_train, X_dev = pre.train_test_split(X)
        y_train, y_dev = pre.train_test_split(y)
        X_train, y_train = pre.shuffle(X_train, y_train)
        X_dev, y_dev = pre.shuffle(X_dev, y_dev)

        def hypermodel(hp):
            return gru.gru_forecaster(
                win = hp.Choice("win", values = self.search_space["win"]),
                size1 = hp.Choice("size1", values = self.search_space["size1"]), 
                dropout1 = hp.Float(
                    "dropout1", 
                    min_value = self.search_space["dropout1"][0],
                    max_value = self.search_space["dropout1"][1],
                    step = 0.5,
                ),
                rec_dropout1 = hp.Float(
                    "rec_dropout1", 
                    min_value = self.search_space["rec_dropout1"][0],
                    max_value = self.search_space["rec_dropout1"][1],
                    step = 0.5,
                ),
                size2 = hp.Choice("size1", values = self.search_space["size2"]),
                dropout2 = hp.Float(
                    "dropout2", 
                    min_value = self.search_space["dropout2"][0],
                    max_value = self.search_space["dropout2"][1],
                    step = 0.5,
                ),
                rec_dropout2 = hp.Float(
                    "rec_dropout2", 
                    min_value = self.search_space["rec_dropout2"][0],
                    max_value = self.search_space["rec_dropout2"][1],
                    step = 0.5,
                ),
            )

        hb = Hyperband(
            hypermodel, factor = 3, seed = 42, overwrite = True,
            objective = Objective("val_loss", direction="min"),
        )
        hb.search(
            X_train, y_train,
            validation_data = (X_dev, y_dev),
            epochs = 20,
            callbacks = [
                callbacks.EarlyStopping(
                    monitor = 'val_loss',
                    patience = 3,
                ), ]
        )
        best_hps = hb.get_best_hyperparameters()[0]
        self.best_config_ = {k: v  for k, v in best_hps.values.items()}
        self.win = self.best_config_["win"]

        forecaster = hb.hypermodel.build(best_hps)
        history = forecaster.fit(
            X_train, y_train, 
            batch_size = 2,
            epochs = 20,
            validation_data = (X_dev, y_dev),
            callbacks = [
                callbacks.EarlyStopping(
                    monitor = 'val_loss',
                    patience = 3,
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
        y_hat = self.forecaster.predict(X_dev).flatten()
        rmse = mean_squared_error(y_dev, y_hat, squared = False)
        forecaster_configs = [{
            "config": self.best_config_,
            "rmse": rmse,
        },]


        wgTuner = WindGaussTuner()
        wgTuner.set_space({
            "w": self.search_space["gauss_win"],
            "step": self.search_space["gauss_step"],
        })
        wgTuner.tune(ts)
        self.classifier = wgTuner.best_model_
        
        self.explored_cfgs_["predictor"] = forecaster_configs
        self.explored_cfgs_["classifier"] = wgTuner.explored_cfgs_

        self.preload = ts[cfg.cols["X"]].copy().to_numpy()[-self.win : ]

        model = gru.GRU()
        self.best_config_["gauss_win"] = wgTuner.best_config_["w"]
        self.best_config_["gauss_step"] = wgTuner.best_config_["step"]
        model.set_params(**self.best_config_)
        model.forecaster = self.forecaster
        model.classifier = self.classifier
        model.preload = self.preload
        self.best_model_ = model
        self.best_score_ = 0
        return self

class LSTMTuner(AbstractTuner):
    def __init__(self):
        super().__init__(search_space = dict(
            win = [26, ], 
            size1 = [64, 128, 256],
            dropout1 = [.3, .5],
            rec_dropout1 = [.3, .5],
            size2 = [64, 128, 256],
            dropout2 = [.3, .5],
            rec_dropout2 = [.3, .5],
            gauss_win = [16, 32, 64, 128],
            gauss_step = [8, 16, 32, 64]
        ))
    
    def tune(self, ts):
        self.explored_cfgs_ = {}
        pre = Preprocessor(ts)
        ts = pre.nan_filled
        # tune forecaster
        X, y = pre.extract_windows(
            ts[cfg.cols["X"]].copy().to_numpy(), 
            w = self.search_space["win"][0]
        )
        X_train, X_dev = pre.train_test_split(X)
        y_train, y_dev = pre.train_test_split(y)
        X_train, y_train = pre.shuffle(X_train, y_train)
        X_dev, y_dev = pre.shuffle(X_dev, y_dev)

        def hypermodel(hp):
            return lstm.lstm_forecaster(
                win = hp.Choice("win", values = self.search_space["win"]),
                size1 = hp.Choice("size1", values = self.search_space["size1"]), 
                dropout1 = hp.Float(
                    "dropout1", 
                    min_value = self.search_space["dropout1"][0],
                    max_value = self.search_space["dropout1"][1],
                    step = 0.5,
                ),
                rec_dropout1 = hp.Float(
                    "rec_dropout1", 
                    min_value = self.search_space["rec_dropout1"][0],
                    max_value = self.search_space["rec_dropout1"][1],
                    step = 0.5,
                ),
                size2 = hp.Choice("size1", values = self.search_space["size2"]),
                dropout2 = hp.Float(
                    "dropout2", 
                    min_value = self.search_space["dropout2"][0],
                    max_value = self.search_space["dropout2"][1],
                    step = 0.5,
                ),
                rec_dropout2 = hp.Float(
                    "rec_dropout2", 
                    min_value = self.search_space["rec_dropout2"][0],
                    max_value = self.search_space["rec_dropout2"][1],
                    step = 0.5,
                ),
            )

        hb = Hyperband(
            hypermodel, factor = 3, seed = 42, overwrite = True,
            objective = Objective("val_loss", direction="min"),
        )
        hb.search(
            X_train, y_train,
            validation_data = (X_dev, y_dev),
            epochs = 20,
            callbacks = [
                callbacks.EarlyStopping(
                    monitor = 'val_loss',
                    patience = 3,
                ), ]
        )
        best_hps = hb.get_best_hyperparameters()[0]
        self.best_config_ = {k: v  for k, v in best_hps.values.items()}
        self.win = self.best_config_["win"]

        forecaster = hb.hypermodel.build(best_hps)
        history = forecaster.fit(
            X_train, y_train, 
            batch_size = 2,
            epochs = 20,
            validation_data = (X_dev, y_dev),
            callbacks = [
                callbacks.EarlyStopping(
                    monitor = 'val_loss',
                    patience = 3,
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
        y_hat = self.forecaster.predict(X_dev).flatten()
        rmse = mean_squared_error(y_dev, y_hat, squared = False)
        forecaster_configs = [{
            "config": self.best_config_,
            "rmse": rmse,
        },]

        wgTuner = WindGaussTuner()
        wgTuner.set_space({
            "w": self.search_space["gauss_win"],
            "step": self.search_space["gauss_step"],
        })
        wgTuner.tune(ts)
        self.classifier = wgTuner.best_model_

        self.explored_cfgs_["predictor"] = forecaster_configs
        self.explored_cfgs_["classifier"] = wgTuner.explored_cfgs_

        self.preload = ts[cfg.cols["X"]].copy().to_numpy()[-self.win : ]

        model = lstm.LSTM()
        self.best_config_["gauss_win"] = wgTuner.best_config_["w"]
        self.best_config_["gauss_step"] = wgTuner.best_config_["step"]
        model.set_params(**self.best_config_)
        model.forecaster = self.forecaster
        model.classifier = self.classifier
        model.preload = self.preload
        self.best_model_ = model
        self.best_score_ = 0
        return self

class SARIMAXTuner(AbstractTuner):
    def __init__(self):
        super().__init__(search_space = dict(
            gauss_win = [16, 32, 64, 128],
            gauss_step = [8, 16, 32, 64]
        ))

    def tune(self, ts):
        self.explored_cfgs_ = {}

        pre = Preprocessor(ts)
        ts = pre.nan_filled
        X = ts
        if cfg.cols["y"] in ts.columns:
            X = ts.drop(cfg.cols["y"], axis = 1)
        self.best_model_ = sarimax.SARIMAX().fit(X)
        #predict: ts[cfg.cols["X"]]
        y_hat = self.best_model_.forecaster.predict().to_numpy()
        score = mean_squared_error(
            X[cfg.cols["X"]].to_numpy(), 
            y_hat, 
            squared = False
        )
        forecaster_cfgs_ = [{
            "config": {
                "order": self.best_model_.order,
                "seasonal_order": self.best_model_.seasonal_order,
            },
            "rmse": score,
        }]
        
        wgTuner = WindGaussTuner()
        wgTuner.set_space({
            "w": self.search_space["gauss_win"],
            "step": self.search_space["gauss_step"],
        })
        wgTuner.tune(ts)
        self.classifier = wgTuner.best_model_
        
        self.explored_cfgs_["predictor"] = forecaster_cfgs_
        self.explored_cfgs_["classifier"] = wgTuner.explored_cfgs_

        self.best_config_ = {}
        self.best_config_["order"] = self.best_model_.order
        self.best_config_["seasonal_order"] = self.best_model_.seasonal_order
        self.best_config_["gauss_win"] = wgTuner.best_config_["w"]
        self.best_config_["gauss_step"] = wgTuner.best_config_["step"]

        self.best_model_.classifier = self.classifier
        self.best_model_.gauss_win = wgTuner.best_config_["w"]
        self.best_model_.gauss_step = wgTuner.best_config_["step"]
        self.best_score_ = 0
        return self

class ProphetTuner(AbstractTuner):
    def __init__(self):
        super().__init__(search_space = dict(
            gauss_win = [16, 32, 64, 128],
            gauss_step = [8, 16, 32, 64],
            changepoint_prior_scale = [0.001, 0.01, 0.1, 0.5],
            seasonality_prior_scale = [0.01, 0.1, 1.0, 10.0],
        ))

    def tune(self, ts):
        self.explored_cfgs_ = {}
        #Gridsearch
        self.forecaster = None
        self.forecaster_config = None
        self.forecaster_score = float('inf')
        forecaster_cfgs_ = []

        prophet_space = {}
        prophet_space["changepoint_prior_scale"] = self.search_space["changepoint_prior_scale"]
        prophet_space["seasonality_prior_scale"] = self.search_space["seasonality_prior_scale"]
        prophet_configs = [dict(zip(prophet_space.keys(), v)) 
                            for v in itertools.product(*prophet_space.values())]
        
        #prophet_ts = pd.DataFrame()
        #prophet_ts[cfg.prophet["timestamp"]] = ts[cfg.cols["timestamp"]]
        #prophet_ts[cfg.prophet["value"]] = ts[cfg.cols["X"]]
        for config in prophet_configs:
            p = prophet.Prophet(
                changepoint_prior_scale = config["changepoint_prior_scale"],
                seasonality_prior_scale = config["seasonality_prior_scale"]
            )
            p.fit(ts)
            future = prophet.make_future_dataframe(
                periods = len(ts), 
                start_date = ts[cfg.cols["timestamp"]].min(),
                freq = "W"
            )
            prophet_res = p.predict(future)
            y_hat = prophet_res["yhat"].to_numpy()

            score = mean_squared_error(
                ts[cfg.cols["X"]].to_numpy(), 
                y_hat,
                squared = False
            )
            forecaster_cfgs_.append({
                "config": config,
                "rmse": score,
            })
            
            if score < self.forecaster_score:
                self.forecaster_score = score
                self.forecaster = p
                self.forecaster_config = config
        
        wgTuner = WindGaussTuner()
        wgTuner.set_space({
            "w": self.search_space["gauss_win"],
            "step": self.search_space["gauss_step"],
        })
        wgTuner.tune(ts)
        self.classifier = wgTuner.best_model_
        
        self.explored_cfgs_["predictor"] = forecaster_cfgs_
        self.explored_cfgs_["classifier"] = wgTuner.explored_cfgs_

        self.best_config_ = {}
        for k, v in self.forecaster_config.items():
            self.best_config_[k] = v
        self.best_config_["gauss_win"] = wgTuner.best_config_["w"]
        self.best_config_["gauss_step"] = wgTuner.best_config_["step"]

        model = prophet.Prophet()
        model.set_params(**self.best_config_)
        model.forecaster = self.forecaster
        model.classifier = self.classifier
        self.best_model_ = model
        self.best_score_ = 0
        return self