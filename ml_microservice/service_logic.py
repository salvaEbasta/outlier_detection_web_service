import os
from importlib import reload
import time
from datetime import date, datetime
from threading import Thread
from multiprocessing import Process
import json
import re
import logging
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from . import constants
from .anomaly_detection import model_factory
from .anomaly_detection import preprocessing
from .anomaly_detection import detector
from .anomaly_detection import metrics

STATUS = dict(active='active', training='under_training')

class DetectorTrainer():
    def __init__(self, 
                    storage_path=constants.detectorTrainer.path,
                    retrain_window=constants.detectorTrainer.retrain_patience,
                ):
        self.logger = logging.getLogger('trainer')
        self.logger.setLevel(logging.DEBUG)

        self.logger.debug("DetectorTrainer: __init__")
        self.storage = storage_path
        self.retrain_window = retrain_window

        self.version_format = constants.formats.version
        self.detector_summary_file = constants.files.detector_summary

        self._detector = None

        self._env = None
        self._identifier = None
        self._version = None

        self._summary = None

    @property
    def detectors_list(self):
        tmp = []
        for d, path in [(f, os.path.join(self.storage, f)) for f in os.listdir(self.storage)]:
            if not os.path.isdir(path):
                continue
            tmp.append(dict(
                model=d, 
                versions=[f for f in os.listdir(os.path.join(self.storage, d))
                            if os.path.isdir(os.path.join(path, f))],
            ))
        return tmp

    def train(self, label: str, training: dict, forecasting_model: str):
        """
            Inputs:\n
            - training: {'label': <-column->, 'dataset': <-dataset->, 'column': <-column->|None }\n
            Steps: Preprocessing -> Training -> Evaluation -> Persistence\n
            Outputs:\n
            {'id', 'version', 'training': {'label', 'dataset', 'column'}, 'forecasting_model', 'training_performance': {}}
        """
        identifier = label
        label = training.get('label', '')
        dataset = training.get('dataset', '')
        column = training.get('column', None)
        # Check and load label, dataset, column
        self.logger.info("[.] Assembling {:s}/{:s}/{}, {:s}".format(label, dataset, column, forecasting_model))
        self.logger.info("[.] Loading dataframe ...")
        dset_lib = TimeseriesLibrary()
        serie, column = dset_lib.fetch_ts(label, dataset, column)

        serie_evaluator = preprocessing.SeriesFilter(None)
        if not serie_evaluator.recent_information_content(serie):
            self.logger.warning("[!] Detected invalid serie: {:s}/{:s}-{}".format(label, dataset, column))
            raise ValueError("Invalid serie, not enough recent information: {:s}/{:s}-{}".format(label, dataset, column))

        # Check forecasting_model
        if not model_factory.ForecasterFactory().has(forecasting_model):
            self.logger.warning("[!] Forecasting model {:s} not found".format(forecasting_model))            
            raise ValueError('The forecasting model {:s} is unavailable'.format(forecasting_model))

        # Preprocessing
        self.logger.info("[.] Composing preprocessor")
        window_size = constants.detectorDefaults.win_size
        train, dev, test = preprocessing.split(serie, window_size=window_size)
        preprocessor = preprocessing.Preprocessor(
            train=train,
            dev=dev,
            test=test,
            input_width=window_size,
        )

        # Check for number of values
        if not serie_evaluator.enough_datapoints(preprocessor.train[0]):
            self.logger.warning(f'[!] The serie has not enough datapoints (extracted: {preprocessor.last_train[0].shape[0]}, needed: {serie_evaluator.min_datapoints}) to train a good anomaly detector')
            raise ValueError(f'The serie has not enough datapoints (extracted: {preprocessor.last_train[0].shape[0]}, needed: {serie_evaluator.min_datapoints}) to train a good anomaly detector')

        # Enviroment setup
        self.logger.info("[.] Building environment")
        self.create_env(identifier)

        summ = Summary(label=label, dataset=dataset, column=column)
        summ.save(self._env)

        # Detector training
        self.logger.info("[*] Training")
        self._detector = detector.Detector(
            window_size=window_size,
            l=constants.detectorDefaults.lambda_,
            k=constants.detectorDefaults.k,
            forecasting_model=forecasting_model,
        )
        tStart = time.time()
        _, _, epochs, history = self._detector.fit(
                                *preprocessor.augmented_train, 
                                dev_data=preprocessor.dev,
                                max_epochs=constants.detectorDefaults.max_epochs,
                                patience=constants.detectorDefaults.early_stopping_patience,
                            )
        tDelta = time.time() - tStart

        # Evaluate regressor
        X_test, y_test = preprocessor.test
        anomalies, y_hat, history = self._detector.detect_update(X_test, y_test)

        prediction_performance = history.naive_score[-1]
        summ._train_time = tDelta
        summ._status = STATUS['active']

        # Save resources
        self.logger.info("[.] Saving resources to {} ...".format(self._env))
        self._detector.save(self._env)
        preprocessor.save_params(self._env)

        summ.save(self._env)
        self.logger.debug("[.] Done")
        return dict(
            id=self._identifier,
            version=self._version,
            training=dict(
                label=label,
                dataset=dataset,
                column=column,
            ),
            forecasting_model=forecasting_model,
            training_performance=dict(
                naive_score=float(prediction_performance),
                y=y_test.tolist(),
                anomalies=anomalies.tolist(),
                total_time=tDelta,
            )
        )

    def create_env(self, identifier: str):
        self._env = os.path.join(self.storage, identifier)
        if not os.path.exists(self._env):
            version = self.version_format % 0
        else:
            version = self.version_format % len([f for f in os.listdir(self._env) 
                                                    if os.path.isdir(os.path.join(self._env, f))])
        self._env = os.path.join(self._env, version)
        self._identifier = identifier
        self._version = version
        os.makedirs(self._env)
        return version

    def retrieve_env(self, identifier: str, version: str):
        self._env = os.path.join(self.storage, identifier)
        if not os.path.exists(self._env):
            self.logger.warning('Can\'t find the identifier \'{:s}\''.format(identifier))
            raise ValueError('Can\'t find the identifier \'{:s}\''.format(identifier))
        self._env = os.path.join(self._env, version)
        if not os.path.exists(self._env):
            self.logger.warning('Can\'t find the specified version, \'{:s}\', under \'{:s}\''.format(version, identifier))
            raise ValueError('Can\'t find the specified version, \'{:s}\', under \'{:s}\''.format(version, identifier))
        self._identifier = identifier
        self._version = version
        self.logger.debug(f"Current env: {self._env}")
        return self._env

    @property
    def loaded(self):
        return self._detector != None

    def load_detector(self, identifier: str, version: str = None):
        """ If version==None: latest recorded best naive performance version\n
            -> version; env
        """
        if version is None:
            env = os.path.join(self.storage, identifier)
            if not os.path.exists(env):
                self.logger.warning('Can\'t find the identifier \'{:s}\''.format(identifier))
                raise ValueError('Can\'t find the identifier \'{:s}\''.format(identifier))
            vs = []
            for d in [f for f in os.listdir(env)
                        if os.path.isdir(os.path.join(env, f)) and 
                            constants.files.detector_summary in os.listdir(os.path.join(env, f))]:
                s = Summary()
                s.load(os.path.join(os.path.join(env, d), constants.files.detector_summary))
                if s.is_active():
                    h = detector.History()
                    h.load(os.path.join(os.path.join(env, d), constants.files.detector_history))
                    vs.append( (d, h.naive_score[-1]) )
            if len(vs) == 0:
                self.logger.warning('No available model for identifier \'{:s}\''.format(identifier))
                raise ValueError('No available model for identifier \'{:s}\''.format(identifier))
            version = max(vs, key = lambda x: x[1] )
            version = version[0]
        return version, self._load_detector(identifier, version)

    def _load_detector(self, identifier: str, version: str):
        self.logger.info("Attempt to load detector {:s}.{:s}".format(identifier, version))
        self.retrieve_env(identifier, version)
        self._summary = Summary()
        self._summary.load(os.path.join(self._env, constants.files.detector_summary))
        self._detector = None
        if self._summary.is_active():
            self._detector = detector.Detector(path=self._env)
            self.logger.info("Successful")
        return self._env
        
    def performance_degradation(self, history):
        self.logger.info("Performance degradation detection")
        naive_score = history.naive_score
        mean = np.mean(naive_score)
        if len(naive_score) < self.retrain_window:
            self.logger.info(f"Not enough info to compute performance degradation (need {self.retrain_window}, have {len(naive_score)})")
            return False
        else:
            self.logger.debug(f"Mean: {mean}, window mean: {np.mean(naive_score[-self.retrain_window:])}")
            return mean < np.mean(naive_score[-self.retrain_window:])
        
    def detect(self, data, store: bool = False, pre_load_data: dict = None):
        """
            Requires a detector to be loaded beforehand\n
            pre_load: \n
                        - None: No padding or preloading\n
                        - {'label': <-label->, 'dataset': <-dataset->, 'column': <-column->}: default is training label-dataset-column\n
        """
        self.logger.info("[.] Requested detection")
        if pre_load_data is not None:
            dl = TimeseriesLibrary()
            pre_load_data, _ = dl.fetch_ts(
                pre_load_data.get('label', self._summary._label),
                pre_load_data.get('dataset', self._summary._dataset),
                pre_load_data.get('column', self._summary._column)
            )
            self.logger.debug("Fetched pre load data")
            data = preprocessing.Padder(
                        serie = np.array(data).flatten(),
                        padding_length = self._detector.window_size,
                    ).pre_loading(as_padding=pre_load_data)
        else:
            data = np.array(data).flatten()
            if data.shape[0] < self._detector.window_size:
                self.logger.warning('Insufficient data to run a detection.' \
                    + ' Got {:d}, need at least {:d}'.format(data.shape[0], self._detector.window_size))
                raise ValueError('Insufficient data to run a detection.' \
                    + ' Got {:d}, need at least {:d}'.format(data.shape[0], self._detector.window_size)) 
        
        # Preprocessing
        self.logger.info("[.] Preprocessor init")
        preproc = preprocessing.Preprocessor(train = data)
        preproc.load_params(os.path.join(self._env, constants.files.preprocessing_params))
        X, y = preproc.train
        
        # Detection
        self.logger.info("[*] Detection ...")
        
        if not store:
            t0 = time.time()
            anomalies, y_hat, history = self._detector.detect(X, y)
            tDelta = time.time() - t0
        else:
            t0 = time.time()
            anomalies, y_hat, history = self._detector.detect_update(X, y)
            tDelta = time.time() - t0
            self._detector.save(self._env)
        self.logger.info(f"[.] Total time: {tDelta}")

        regression_performance = history.naive_score[-1]
        self.logger.info("[.] Overall regression performance on data: {:.6f}".format(regression_performance))

        degradation = False
        # Retrain decision
        if store and self.performance_degradation(history):
            self.logger.info("[*] Detected degradation, retraining")
            degradation = True
            trainT = Thread(
                target = self.train, 
                args = (
                    self._identifier,
                    self._summary.training,
                    self._detector._forecasting_model
                )
            )
            trainT.start()
        return dict(
            data = data.tolist(),
            start_detection_idx = self._detector.window_size,
            anomalies = anomalies.tolist(),
            rmse = history.rmse[-len(y):].tolist(),
            naive_score = history.naive_score[-len(y):].tolist(),
            total_time = tDelta,
            degradation = 'not_evaluated' if not store else str(degradation),
        )

class TimeseriesLibrary:
    def __init__(self, 
        path=constants.timeseries.path, 
        date_col=constants.timeseries.date_column
    ):
        self.logger = logging.getLogger('tsLib')
        self.logger.setLevel(logging.DEBUG)
        self.storage = path
        self.date_col = date_col

    @property
    def timeseries(self):
        """
        -> [{group: str, dimensions: [str, ...]}, ...]
        """
        groups = []
        for f in os.listdir(self.storage):
            abs_f = os.path.join(self.storage, f)
            if os.path.isdir(abs_f):
                groups.append({
                    "group": f,
                    "dimensions": [
                        csv[:-4] for csv in os.listdir(abs_f)
                            if re.match('.*.csv', csv) is not None
                    ],
                })
        return groups

    def _2storage_path(self, group :str, dim: str = None):
        """-> (path: str, exists: bool)"""
        stg_path = os.path.join(self.storage, group)
        if dim is None:
            return stg_path, os.path.exists(stg_path)
        stg_path = os.path.join(stg_path, "{:s}.csv".format(dim))
        return stg_path, os.path.exists(stg_path)

    def fetch(self, group :str, dim: str):
        """ -> pandas.DataFrame """
        self.logger.info("Fetch {}/{}".format(group, dim))
        df = None
        if self.has(group, dim):
            df = os.path.join(self.storage, group, "{:s}.csv".format(dim))
            df = pd.read_csv(df)
            if 'Unnamed: 0' in df.columns:
                df = df.drop('Unnamed: 0', axis=1)
            df[self.date_col] = pd.to_datetime(df[self.date_col])
        return df

    def fetch_ts(self, group: str, dim: str, tsID: str):
        """ -> pd.DataFrame"""
        self.logger.info("[.] Fetch ts: {:s}/{:s}-{:s}".format(group, dim, tsID))
        if not self.has_group(group):
            self.logger.warning("[!] Group \'{:s}\' not found".format(group))
            raise ValueError('Group \'{:s}\' not found'.format(group))
        if not self.has(group, dim):
            self.logger.warning("[!] Dimension \'{:s}/{:s}\' not found".format(group, dim))
            raise ValueError('The dimension \'{:s}\' not in group \'{:s}\''.format(dim, group))
        df = self.fetch(group, dim)
        if tsID not in df.columns:
            self.logger.warning("[!] tsID {:s} not in {:s}/{:s}".format(tsID, group, dim))
            raise ValueError('tsID {:s} not in {:s}/{:s}'.format(tsID, group, dim))
        df = df[[self.date_col, tsID]]
        return df, tsID

    def has_group(self, group: str):
        return self._2storage_path(group)[-1]

    def has_dimension(self, dim: str):
        for j in self.timeseries:
            for d in j["dimensions"]:
                if dim == d:
                    return True
        return False

    def has(self, group:str, dimension: str):
        return self._2storage_path(group, dimension)[-1]

    def remove(self, group: str, dimension: str = None, tsID: str = None):
        if dimension is None and tsID is not None:
            logging.warning("[!] Invalid removal, tsID specified when dimension is not")
            return False
        df_path, is_real = self._2storage_path(group)
        if not is_real:
            return True
        df_path, is_real = self._2storage_path(group, dimension)
        if not is_real:
            return True
        if tsID is None:
            os.remove(df_path)
            return True
        if tsID == self.date_col:
            logging.warning("[!] Asked to remove date ts, refused")
            return False
        df = pd.read_csv(df_path)
        if tsID not in df.columns:
            return True
        df = df.drop(tsID, axis=1)
        os.remove(df_path)
        df.to_csv(df_path, index=False)
        return True

    def save(self, group: str, dfID: str, df: pd.DataFrame, override: bool = False):
        """ No merging """
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        if not override and self.has(group, dfID):
            old_df = self.fetch(group, dfID)
            cols = set()
            [cols.add(c) for c in old_df.columns]
            [cols.add(c) for c in df.columns]
            for c in cols:
                if c not in old_df:
                    old_df[c] = [np.nan]*len(old_df)
                if c not in df:
                    df[c] = [np.nan]*len(df)
            for _, row in df.iterrows():
                last_date = old_df.iloc[-1][self.date_col]
                if (
                    last_date.year == row[self.date_col].year and 
                    last_date.month == row[self.date_col].month and
                    last_date.day == row[self.date_col].day
                ):
                    continue
                old_df = old_df.append(row, ignore_index=True)
            df = old_df
        self.remove(group, dfID)
        df_path = self._2storage_path(group, dfID)    
        df.to_csv(df_path, index=False)
        return True

class Summary():
    def __init__(self, 
                    label = None, 
                    dataset = None, 
                    column = None,
                    status = STATUS["training"], 
                    created_on = datetime.now().isoformat(),
                    train_time = -1,
                ):
        self._status = status
        self._created_on = created_on

        self._train_time = train_time
        self._label = label
        self._dataset = dataset
        self._column = column

    @property
    def training(self):
        return dict(
            total_time = self._train_time,
            label = self._label,
            dataset = self._dataset,
            column = self._column,
        )

    @property
    def values(self):
        return dict(
                status = self._status,
                created_on = self._created_on,
                training = self.training,
            )

    def is_active(self):
        return self._status == STATUS["active"]

    def save(self, ddir):
        f = os.path.join(ddir, constants.files.detector_summary)
        with open(f, 'w') as f:
            json.dump(self.values, f)

    def load(self, path):
        if not os.path.exists(path):
            self.logger.warning('Can\'t find Summary @{}'.format(path))
            self.__init__()
        else:
            with open(path, 'r') as f:
                tmp = json.load(f)
            self.__init__(
                label = tmp['training']['label'],
                dataset = tmp['training']['dataset'],
                column = tmp['training']['column'],
                train_time = tmp['training']['total_time'],
                status = tmp['status'],
                created_on = tmp['created_on']
            )

    def __repr__(self):
        return "Summary(status: {}, created_on: {}, training: {})".format(
            self._status,
            self._created_on,
            self.training,
        )
