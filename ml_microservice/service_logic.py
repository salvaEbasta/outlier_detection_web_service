import os
from importlib import reload
import configparser
import time
from datetime import datetime
from multiprocessing import Process
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from . import strings
from .anomaly_detection import model_factory
from .anomaly_detection import preprocessing
from .anomaly_detection import detector
from .anomaly_detection import metrics

conf = configparser.ConfigParser()
conf.read(strings.config_file)

STATUS = dict(active='active', training='under training')

class DetectorsLibrary():
    def __init__(self, storage_path=conf['Series']['path']):
        self.storage = storage_path
        self.version_format = strings.version_format
        self.detector_summary = strings.model_summary_file

    @property
    def list(self):
        labels = []
        for d in os.listdir(self.storage):
            label = dict()
            label["versions"] = os.listdir(os.path.join(self.storage, d))
            label["serie"] = d
            labels.append(label)
        return labels

    def assemble(self, blueprint: dict):
        """Check and assemble (preprocess -> train -> evaluate)\n
            blueprint: {'label': <-label->, 'regressor': <-regressor_model->, 'dataset': <-dataset->, 
            ['column': <-column_name->],['epochs': <-epochs#->]}
        """
        # Integrity of blueprint
        if blueprint.get('label', None) is None:
            raise ValueError('The blueprint must specify a label')
        elif blueprint.get('regressor', None) is None:
            raise ValueError('The blueprint must specify a regressor')
        elif blueprint.get('dataset', None) is None:
            raise ValueError('The blueprint must specify a dataset')
        if 'epochs' not in blueprint:
            blueprint['epochs'] = 1
        # Enviroment setup
        env = os.path.join(self.storage, blueprint['label'])
        if not os.path.exists(env):
            os.makedirs(env)
        version = self.version_format % len(os.listdir(env))
        print(f"[*] Preparing environment for {blueprint['label']}/{version}")
        env = os.path.join(env, version)
        os.makedirs(env)

        summary = dict(
                        status=STATUS["training"], 
                        created_on=datetime.now().isoformat(),
                        training_time='', 
                        regressor=blueprint['regressor'],
                        dataset=blueprint['dataset'],
                        column='',
                        regression_performance=0,
                        epochs=blueprint['epochs'],
                    )
        with open(os.path.join(env, self.detector_summary), 'w') as f:
            json.dump(summary, f)

        # Dataset loading
        library = DatasetsLibrary()
        if not library.has(blueprint['dataset']):
            raise ValueError(f'The dataset {blueprint["dataset"]} is not available')
        print(f"[*] Loading dataset {blueprint['dataset']}")
        dframe = library.fetch(blueprint['dataset'])
        if blueprint.get('column', None) is None:
            blueprint['column'] = dframe.columns[1] # Default value
        summary['column'] = blueprint['column']
        dframe = dframe[blueprint['column']]

        # Preprocessing
        print(f"[*] Preprocessor loading ...")
        # TODO: dividere già in precedenza fra train e test, ho un modo per valutare le prestazioni
        train, test = preprocessing.split(dframe, dev=False)
        preprocessor = preprocessing.Preprocessor(
            train=train,
            test=test,
            input_width=int(conf['AnomalyDetector']['window']),
        )

        # Detector training
        anomaly_detector = detector.Detector(
            window_size=int(conf['AnomalyDetector']['window']),
            l=float(conf['AnomalyDetector']['lambda']),
            k=float(conf['AnomalyDetector']['k']),
            regressor_model=blueprint['regressor'],
        )
        print(f"[*] Training ...")
        tStart = time.time()
        anomaly_detector.fit(*preprocessor.train, epochs=blueprint['epochs'])
        tDelta = time.time() - tStart

        # Evaluate regressor
        X_test, y_test = preprocessor.test
        y_hat = anomaly_detector.predict(X_test)
        reg_performance = metrics.naive_model_metric(X_test, y_test, y_hat)
        summary['regression_performance'] = float(reg_performance)
        summary['training_time'] = tDelta
        summary['status'] = STATUS['active']

        # Save detector
        anomaly_detector.save(env)

        # Dump summary
        print("[.] Dumping summary")
        summary.pop('regressor', None)
        with open(os.path.join(env, self.detector_summary), 'w') as f:
            json.dump(summary, f)
        summary['regressor'] = blueprint['regressor']
        return summary

class AnomalyDetection():
    def __init__(self, path=conf['Series']['path'], **description):
        self.storage = path
        if 'label' not in description:
            raise ValueError('A label must be specified in order to load the correct detector')
        label_path = os.path.join(self.storage, description.get('label'))
        
        # Version integrity
        if description.get('version', None) is None:
            versions = os.listdir(label_path)
            versions.sort()
            description['version'] = versions[-1]
        elif description.get('version') not in os.listdir(label_path):
            raise ValueError('The specified version \'{:s}\' is not available'.format(description['version']))
        
        # Init AnomalyDetector
        self._env = os.path.join(label_path, description.get('version'))
        self._detector = None
        self._check_status_and_load()

    @property
    def info(self):
        with open(os.path.join(self._env, strings.model_summary_file), 'r') as f:
            self._summary = json.load(f)
        if self._check_status_and_load():
            return self._detector.params | self._summary
        else:
            return self._summary

    @property
    def detector_ready(self) -> bool:
        return self.detector_status is not STATUS['training']

    def _check_status_and_load(self):
        ready = self.detector_ready
        if ready:
            self._detector = detector.Detector(path=self._env)
        else:
            self._detector = None
        return ready
        
    @property
    def detector_status(self) -> bool:
        with open(os.path.join(self._env, strings.model_summary_file), 'r') as f:
            status = json.load(f)['status']
        return status

    def _predict(self, data):
        data = np.array(data)
        print(f"[.] Prediction for {self._env}")
        print(f"[.] Data: {data}")
        assert len(data.shape) == 1

        info = self.info
        dataset, column = info['dataset'], info['column']

        # Preprocessing data
        print("[*] Initializing preprocessor ...")
        dsetLib = DatasetsLibrary()
        dframe = dsetLib.fetch(dataset)
        dframe = dframe[column]
        train, _ = preprocessing.split(dframe, dev=False)
        preprocessor = preprocessing.Preprocessor(
            train=train,
            test=data, 
            input_width=self._detector.window_size,
        )
        X, y = preprocessor.test
        
        # Detection
        print(f"[*] Detection ...")
        t0 = time.time()
        anomalies = self._detector.detect(X, y)
        tDelta = time.time() - t0
        print(f"[.] Total time: {tDelta}")

        regression_performance = metrics.naive_model_metric(X, y, self._detector.predict(X))
        print("[.] Overall regression performance on data: {:.6f}".format(regression_performance))
        return dict(
            not_evaluated_until=self._detector.window_size - 1, 
            anomalies=anomalies.tolist(), 
            total_time=tDelta,
        )
        

    def predict(self, data):
        """ 
            Check for anomalies in a data serie in input\n
            input: list ( D x 1 )\n
            output: list of anomalies\n
            Assumption: the data comes from the <dataset>, <column> on which the model was originally trained\n
        """
        if self._check_status_and_load():
            return self._predict(data)
        else:
            return dict(
                not_evaluated_until=-1,
                anomalies=[],
                total_time=0,
            )

    def _update(self, data, epochs=1):
        data = np.array(data)
        assert len(data.shape) == 1
        print(f"[.] Update for {self._env}")
        print(f"[.] New data: {data}")

        info = self.info
        dataset, column = info['dataset'], info['column']

        # Merge datasets : [train|dev|test]|[data] -> [train  |dev|test]
        # TODO: save the new data in the dataset -> trailing procedure
        # dsetLib.extend(<dataset>, <new_data>, [<new_dataset>])
        # dsetLib.fetch(<dataset>)
        print("[*] Merging old and new data ... ")        
        dsetLib = DatasetsLibrary()
        dframe = dsetLib.fetch(dataset)
        old_dframe = dframe[column]
        dframe = np.concatenate((np.array(old_dframe), data), axis=0)
        
        # Preprocessing data
        print("[*] Initializing preprocessor ...")
        old_train, _ = preprocessing.split(old_dframe, dev=False)
        train, test = preprocessing.split(dframe, dev=False)

        # Removal of already seen istances
        train = train[len(old_train) - self._detector.window_size:]
        preprocessor = preprocessing.Preprocessor(
            train=train,
            test=test, 
            input_width=self._detector.window_size,
        )
        X, y = preprocessor.test

        # Detector training
        print(f"[*] Training with new data ...")
        t0 = time.time()
        self._detector.fit(*preprocessor.train)
        tDelta = time.time() - t0
        print(f"[.] Total time: {tDelta}")

        # Evaluate regressor
        X_test, y_test = preprocessor.test
        y_hat = self._detector.predict(X_test)
        regression_performance = metrics.naive_model_metric(X, y, self._detector.predict(X))
        print("[.] Overall regression performance on data: {:.6f}".format(regression_performance))

        # Store model
        self._detector.save(self._env)
        return dict(
            new_data_points=len(train),
            training_time=tDelta,
            regression_performance=float(regression_performance),
        )

    def update(self, data, epochs=1):
        """
            Use the input data to fit the model (already fitted on a dataset).
            Introduce additional knowledge.\n
            Assumption:\n
            a) the data comes from the <dataset>, <column> on which the model was originally trained\n
            b) the incoming first timestamp in data follows the last timestamp in <dataset>, <column>\n
        """
        if self._check_status_and_load():
            return self._update(data, epochs=epochs)
        else:
            return dict(
                new_data_points=0,
                training_time=0,
                regression_performance=0,
                epochs=epochs,
            )

class DatasetsLibrary:
    def __init__(self, path=conf['Datasets']['path']):
        self.location = path

    @property
    def datasets(self):
        result = getattr(self, '_datasets', None)
        if result is None:
            result = os.listdir(self.location)
            self._datasets = result
        return result

    def fetch(self, name: str):
        dataset = None
        if self.has(name):
            return pd.read_csv(os.path.join(self.location, name))
        else:
            return None

    def has(self, dset: str):
        return dset in self.datasets

    def append_store(self, dataset: str, column: str, data, new_dataset: str):
        return self.store(self.append(dataset, column, data), new_dataset)

    def append(self, dataset: str, column: str, data):
        dset = self.fetch(dataset)
        if dset is None:
            return None
        elif column not in dset.columns:
            return None
        old_data = np.array(dset[column])
        return np.concatenate((old_data, data), axis=0)

    def store(self, data, name: str, sep=";"):
        """
            data: numpy array
        """
        if data is not None:
            pd.Dataframe(data).to_csv(name, sep=sep)
        return data
            