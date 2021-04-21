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

conf = configparser.ConfigParser()
conf.read('config.ini')

STATUS = dict(old='decommisioned', active='active', training='not ready')

class Toolshed():
    def __init__(self, storage_path=conf['Series']['path']):
        self.storage = storage_path
        self.version_format = strings.version_format
        self.model_summary = strings.model_summary_file

    @property
    def tools(self):
        return [m for m in os.listdir(self.storage)
                if os.path.isdir(os.path.join(self.storage, m))]

    def assemble(self, blueprint: dict):
        """Check and assemble"""
        if blueprint['label'] in self.tools:
            raise ValueError('Label already in use')

        path = os.path.join(self.storage, blueprint['label'], self.version_format%0)
        print(f"[.] Assembling {blueprint['label']}/{self.version_format%0}")
        if not os.path.exists(path):
            os.makedirs(path)
        summary = dict(status=STATUS["training"], created_on=datetime.now().isoformat(), 
                        architecture=None, model=blueprint['model'], datasets=blueprint['datasets'])
        with open(os.path.join(path, self.model_summary), 'w') as f:
            json.dump(summary, f)
        #p = Process(target=Model.train_store, args=(path, blueprint['datasets']))
        #p.start()
        return summary

    def pickup(self, description: dict):
        return Model(os.path.join(self.storage, description['name']))

class Model():
    def __init__(self, path):
        self.location = path
        self.model = keras.models.load_model(self.location)

    @property
    def info(self):
        info = "test"
        return dict(status='test', architecture=info)

    def predict(self, data):
        X = np.array(data)
        t0 = time.time()
        y = self.model.predict(X)
        delta = time.time() - t0
        return dict(time=delta, prediction=y.tolist())

    def retrain(self):
        raise NotImplementedError('Model:retrain not implemented')

    @staticmethod
    def train_store(path, datasets):
        return None

class DatasetsLibrary:
    def __init__(self, path=conf['Resources']['datasetsPath']):
        self.location = path

    @property
    def datasets(self):
        result = getattr(self, '_datasets', None)
        if result is None:
            result = os.listdir(self.location)
            self._datasets = result
        return result

    def fetch(self, name):
        dataset = None
        if name in self.datasets:
            return pd.read_csv(os.path.join(self.location, name))
        else:
            return None