import configparser
from typing import List, Dict

import tensorflow as tf
from tensorflow import keras

from ml_microservice import strings

conf = configparser.ConfigParser()
conf.read(strings.config_file)

def compileModel(model):
    model.compile(loss="mse")
    return model

def testModel(input_size=10):
    i = keras.Input(shape=(input_size,))
    o = keras.layers.Dense(1, activation='relu')(i)
    test = keras.Model(inputs=i, outputs=o)
    return compileModel(test)

def linearModel(input_size=50):
    i = keras.Input(shape=(input_size,))
    x = keras.layers.Dense(units=10, activation='relu')(i)
    o = keras.layers.Dense(units=1, activation='relu')(x)
    model = keras.Model(inputs=i, outputs=o)
    return compileModel(model)

def lstmModel(input_size=50):
    lstm_l2 = keras.models.Sequential([
                               keras.layers.Reshape((input_size, 1), input_shape=[input_size,]),
                               keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1], dropout=0.25, recurrent_dropout=0.25),
                               keras.layers.LSTM(20, dropout=0.25, recurrent_dropout=0.25),
                               keras.layers.Dense(1),
    ])
    return compileModel(lstm_l2)

def gruModel(input_size=50):
    gru_l2 = keras.models.Sequential([
                               keras.layers.Reshape((input_size, 1), input_shape=[input_size,]),
                               keras.layers.GRU(20, return_sequences=True, input_shape=[None, 1], dropout=0.25, recurrent_dropout=0.25),
                               keras.layers.GRU(20, dropout=0.25, recurrent_dropout=0.25),
                               keras.layers.Dense(1),
    ])
    return compileModel(gru_l2)

def convModel(input_size=50):
    pure_cnn = keras.models.Sequential([
                                    keras.layers.Reshape((input_size, 1), input_shape=[input_size,]),
                                    keras.layers.Conv1D(
                                        filters=16, 
                                        kernel_size=7,
                                        activation='relu',
                                        padding='same',
                                      ),
                                    keras.layers.MaxPool1D(
                                        pool_size=3,
                                        strides=2),
                                    keras.layers.Conv1D(
                                        filters=32,
                                        kernel_size=3,
                                        activation='relu',
                                        padding='same'
                                    ),
                                    keras.layers.Flatten(),
                                    keras.layers.Dense(16, activation='relu'),
                                    keras.layers.Dropout(0.25),
                                    keras.layers.Dense(1)
    ])
    return compileModel(pure_cnn)

def mistoModel(input_size=50):
    misto = keras.models.Sequential([
                                    keras.layers.Reshape((input_size, 1), input_shape=[input_size,]),
                                    keras.layers.Conv1D(
                                        filters=16, 
                                        kernel_size=7,
                                        activation='relu',
                                        padding='same',
                                      ),
                                    keras.layers.MaxPool1D(
                                        pool_size=3,
                                        strides=2),
                                    keras.layers.Conv1D(
                                        filters=32,
                                        kernel_size=3,
                                        activation='relu',
                                        padding='same'
                                    ),
                                    keras.layers.GRU(16, dropout=0.25),
                                    keras.layers.Dense(1)
    ])
    return compileModel(misto)

class RegressorFactory:
    def __init__(self, input_size=int(conf['AnomalyDetector']['window'])):
        self._input_size = input_size

        self.factory = dict()
        self.factory['test'] = testModel
        self.factory['linear'] = linearModel
        self.factory['gru'] = gruModel
        self.factory['lstm'] = lstmModel
        self.factory['cnn'] = convModel
        self.factory['hybrid'] = mistoModel

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