import configparser
from typing import List, Dict

import tensorflow as tf
from tensorflow import keras

from ml_microservice import strings

conf = configparser.ConfigParser()
conf.read(strings.config_file)

def testModel(input_size=10):
    i = keras.Input(shape=(input_size,))
    o = keras.layers.Dense(1, activation='relu')(i)
    test = keras.Model(inputs=i, outputs=o)
    test.compile(loss='mse')
    return test

def linearModel(input_size=50):
    i = keras.Input(shape=(input_size,))
    x = keras.layers.Dense(units=10, activation='relu')(i)
    o = keras.layers.Dense(units=1, activation='relu')(x)
    model = keras.Model(inputs=i, outputs=o)
    model.compile(loss="mse")
    return model

class RegressorFactory:
    def __init__(self, input_size=int(conf['AnomalyDetector']['window'])):
        self._input_size = input_size

        self.factory = dict()
        self.factory['test'] = testModel
        self.factory['linear'] = linearModel

    def available(self) -> List[Dict]:
        if getattr(self, '_summary', None) is None:
            self._summary = []
            for model in self.factory.keys():
                self._summary.append(
                    dict(
                        name=model, 
                        architecture=self.factory[model](self.input_size).to_json()
                    )
                )
        return self._summary
    
    def architecture(self, model: str):
        if model in self.factory.keys():
            return self.factory[model](self.input_size).to_json()
        return None
    
    def has(self, model:str):
        return model in self.factory.keys()

    @property
    def input_size(self):
        return self._input_size

    def build(self, label: str):
        result = None
        if self.has(label):
            result = self.factory[label](self.input_size)
        return result