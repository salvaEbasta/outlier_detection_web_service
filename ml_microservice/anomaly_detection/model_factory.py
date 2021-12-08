from typing import List, Dict

import tensorflow as tf
from tensorflow import keras

from ml_microservice import constants

class HyperModel():
    def __init__(self, builder_id):
        self._id = builder_id

    @property
    def builder_id(self):
        return self._id

    def build(self, window_size):
        raise NotImplementedError()

class HyperTest(HyperModel):
    def __init__(self):
        super().__init__('test')
        self.lr = [1e-1, 5e-2]

    def build(self, window_size):
        def builder(hp):
            i = keras.Input(shape=(window_size,))
            o = keras.layers.Dense(1, activation='relu')(i)
            test = keras.Model(inputs=i, outputs=o)
            test.compile(
                optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=self.lr)),
                loss=keras.losses.MeanSquaredError()
            )
            return test
        return builder

    def __repr__(self):
        return "[Input(), Dense(units: 1, activation: relu), ]"

class HyperLinear(HyperModel):
    def __init__(self):
        super().__init__('linear')
        self.lr = [5e-2, 1e-2]

    def build(self, window_size):
        def builder(hp):
            i = keras.Input(shape=(window_size,))
            x = keras.layers.Dense(units=10, activation='relu')(i)
            o = keras.layers.Dense(units=1, activation='relu')(x)
            model = keras.Model(inputs=i, outputs=o)
            model.compile(
                optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=self.lr)),
                loss=keras.losses.MeanSquaredError()
            )
            return model
        return builder

    def __repr__(self):
        return "[Input(), Dense(units: 10, activation: relu), " \
            + "Dense(units: 1, activation: relu), ]"

class HyperGru(HyperModel):
    def __init__(self):
        super().__init__('gru')
        self.gru0 = (16, 16, 168)
        self.gru0_drop = (0.2, 0.6)
        self.gru0_rec_drop = (0.2, 0.6)

        self.gru1 = (16, 16, 168)
        self.gru1_drop = (0.2, 0.6)
        self.gru1_rec_drop = (0.2, 0.6)
        
        self.lr = [5e-2, 1e-2]

    def build(self, window_size):
        def builder(hp):
            gru_l2 = keras.models.Sequential([
                                    keras.layers.Reshape((window_size, 1), input_shape=[window_size,]),
                                    keras.layers.GRU(hp.Int('gru_0', min_value=self.gru0[0], max_value=self.gru0[2], step=self.gru0[1]),
                                                        return_sequences=True,
                                                        input_shape=[None, 1],
                                                        dropout=hp.Float('gru_0_drop', min_value=self.gru0_drop[0], max_value=self.gru0_drop[1]),
                                                        recurrent_dropout=hp.Float('gru_0_rec_drop', min_value=self.gru0_rec_drop[0], max_value=self.gru0_rec_drop[1]),
                                                        ),
                                    keras.layers.GRU(hp.Int('gru_1', min_value=self.gru1[0], max_value=self.gru1[2], step=self.gru1[1]),
                                                        dropout=hp.Float('gru_1_drop', min_value=self.gru1_drop[0], max_value=self.gru1_drop[1]),
                                                        recurrent_dropout=hp.Float('gru_1_rec_drop', min_value=self.gru1_rec_drop[0], max_value=self.gru1_rec_drop[1]),
                                                        ),
                                    keras.layers.Dense(1),
            ])
            gru_l2.compile(
                    optimizer=keras.optimizers.Adam(
                        hp.Choice('learning_rate', values=self.lr)
                    ),
                    loss=keras.losses.MeanSquaredError()
            )
            return gru_l2
        return builder
    
    def __repr__(self):
        tmp = "[Input(), "
        tmp += "GRU(units:[{:d}:{:d}:{:d}], dropout:[{:.2f}-{:.2f}], recurrent-dropout:[{:.2f}-{:.2f}]), ".format(
                *self.gru0,
                *self.gru0_drop, 
                *self.gru0_rec_drop,
            )
        tmp += "GRU(units:[{:d}:{:d}:{:d}], dropout:[{:.2f}-{:.2f}], recurrent-dropout:[{:.2f}-{:.2f}]), ".format(
                *self.gru1,                 
                *self.gru1_drop, 
                *self.gru1_rec_drop,
            )
        tmp += "Dense(units: 1), ]"
        return tmp

class HyperLstm(HyperModel):
    def __init__(self):
        super().__init__('lstm')
        self.lstm0 = (16, 16, 168)
        self.lstm0_drop = (0.2, 0.6)
        self.lstm0_rec_drop = (0.2, 0.6)

        self.lstm1 = (16, 16, 168)
        self.lstm1_drop = (0.2, 0.6)
        self.lstm1_rec_drop = (0.2, 0.6)
        
        self.lr = [5e-2, 1e-2]

    def build(self, window_size):
        def builder(hp):
            lstm_l2 = keras.models.Sequential([
                                    keras.layers.Reshape((window_size, 1), input_shape=[window_size,]),
                                    keras.layers.LSTM(hp.Int('lstm_0', min_value=self.lstm0[0], max_value=self.lstm0[2], step=self.lstm0[1]),
                                                        return_sequences=True, 
                                                        input_shape=[None, 1], 
                                                        dropout=hp.Float('lstm_0_drop', min_value=self.lstm0_drop[0], max_value=self.lstm0_drop[1]),
                                                        recurrent_dropout=hp.Float('lstm_0_rec_drop', self.lstm0_rec_drop[0], max_value=self.lstm0_rec_drop[1]),
                                                        ),
                                    keras.layers.LSTM(hp.Int('lstm_1', min_value=self.lstm1[0], max_value=self.lstm1[2], step=self.lstm1[1]),
                                                        dropout=hp.Float('lstm_1_drop', min_value=self.lstm1_drop[0], max_value=self.lstm1_drop[1]),
                                                        recurrent_dropout=hp.Float('lstm_1_rec_drop', min_value=self.lstm1_rec_drop[0], max_value=self.lstm1_rec_drop[1]),
                                                        ),
                                    keras.layers.Dense(1),
            ])
            lstm_l2.compile(
                    optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=self.lr)),
                    loss=keras.losses.MeanSquaredError()
            )
            return lstm_l2
        return builder
    
    def __repr__(self):
        tmp = "[Input(), "
        tmp += "LSTM(units:[{:d}:{:d}:{:d}], dropout:[{:.2f}-{:.2f}], recurrent-dropout:[{:.2f}-{:.2f}]), ".format(
                *self.lstm0,
                *self.lstm0_drop, 
                *self.lstm0_rec_drop,
            )
        tmp += "LSTM(units:[{:d}:{:d}:{:d}], dropout:[{:.2f}-{:.2f}], recurrent-dropout:[{:.2f}-{:.2f}]), ".format(
                *self.lstm1,                 
                *self.lstm1_drop, 
                *self.lstm1_rec_drop,
            )
        tmp += "Dense(units: 1), ]"
        return tmp

class HyperCNN(HyperModel):
    def __init__(self):
        super().__init__('cnn')
        self.cnn1d_0_filters = (16, 16, 64)
        self.cnn1d_1_filters = (16, 16, 128)
        self.dense = (8, 8, 32)
        self.dropout = (0.2, 0.6)

        self.lr = [5e-2, 1e-2]

    def build(self, window_size):
        def builder(hp):
            pure_cnn = keras.models.Sequential([
                                            keras.layers.Reshape((window_size, 1), input_shape=[window_size,]),
                                            keras.layers.Conv1D(
                                                filters=hp.Int('cnn1d_0_filters', min_value=self.cnn1d_0_filters[0], max_value=self.cnn1d_0_filters[2], step=self.cnn1d_0_filters[1]), 
                                                kernel_size=7,
                                                activation='relu',
                                                padding='same',
                                            ),
                                            keras.layers.MaxPool1D(
                                                pool_size=3,
                                                strides=2),
                                            keras.layers.Conv1D(
                                                filters=hp.Int('cnn1d_1_filters', min_value=self.cnn1d_1_filters[0], max_value=self.cnn1d_1_filters[2], step=self.cnn1d_1_filters[1]), 
                                                kernel_size=3,
                                                activation='relu',
                                                padding='same'
                                            ),
                                            keras.layers.Flatten(),
                                            keras.layers.Dense(hp.Int('dense', min_value=self.dense[0], max_value=self.dense[2], step=self.dense[1]), 
                                                            activation='relu'),
                                            keras.layers.Dropout(hp.Float('dropout', min_value=self.dropout[0], max_value=self.dropout[1])),
                                            keras.layers.Dense(1)
            ])
            pure_cnn.compile(
                    optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-1, 5e-2, 1e-2])),
                    loss=keras.losses.MeanSquaredError()
            )
            return pure_cnn
        return builder
    
    def __repr__(self):
        tmp = "[Input(), "
        tmp += "Conv1D(filters:[{:d}:{:d}:{:d}], activation: relu, kernel: 7, activation: relu, padding: same), ".format(
                *self.cnn1d_0_filters,
            )
        tmp += "MaxPool1D(pool_size: 3, strides: 2), "
        tmp += "Conv1D(filters:[{:d}:{:d}:{:d}], activation: relu, kernel: 3, activation: relu, padding: same), ".format(
                *self.cnn1d_1_filters,
            )
        tmp += "Flatten(), Dense(units: [{:d}:{:d}:{:d}], activation: relu), Dropout([{:.2f}-{:.2f}]), Dense(units: 1), ]".format(
            *self.dense,
            *self.dropout,
        )
        return tmp

class HyperHybrid(HyperModel):
    def __init__(self):
        super().__init__('hybrid')
        self.cnn0_filters = (16, 16, 64)
        self.cnn1_filters = (16, 16, 128)

        self.gru = (16, 16, 128)
        self.gru_drop = (0.2, 0.6)

        self.lr = [5e-2, 1e-2]

    def build(self, window_size):
        def builder(hp):
            misto = keras.models.Sequential([
                                            keras.layers.Reshape((window_size, 1), input_shape=[window_size,]),
                                            keras.layers.Conv1D(
                                                filters=hp.Int('hybrid_cnn_0_filters', min_value=self.cnn0_filters[0], max_value=self.cnn0_filters[2], step=self.cnn0_filters[1]), 
                                                kernel_size=7,
                                                activation='relu',
                                                padding='same',
                                            ),
                                            keras.layers.Conv1D(
                                                filters=hp.Int('hybrid_cnn_1_filters', min_value=self.cnn1_filters[0], max_value=self.cnn1_filters[2], step=self.cnn1_filters[1]), 
                                                kernel_size=3,
                                                activation='relu',
                                                padding='same'
                                            ),
                                            keras.layers.GRU(hp.Int('hybrid_dense', min_value=self.gru[0], max_value=self.gru[2], step=self.gru[1]), 
                                                            dropout=hp.Float('hybrid_dense_dropout', min_value=self.gru_drop[0], max_value=self.gru_drop[1]),
                                                            ),
                                            keras.layers.Dense(1)
            ])
            misto.compile(
                    optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=super.lr)),
                    loss=keras.losses.MeanSquaredError()
            )
            return misto
        return builder

    def __repr__(self):
        tmp = "[Input(), "
        tmp += "Conv1D(filters:[{:d}:{:d}:{:d}], activation: relu, kernel: 7, activation: relu, padding: same), ".format(
                *self.cnn0_filters,
            )
        tmp += "Conv1D(filters:[{:d}:{:d}:{:d}], activation: relu, kernel: 3, activation: relu, padding: same), ".format(
                *self.cnn1_filters,
            )
        tmp += "GRU(units:[{:d}:{:d}:{:d}], dropout:[{:.2f}-{:.2f}]), ".format(
                *self.gru,
                *self.gru_drop,
            )
        tmp += "Dense(units: 1), ]"
        return tmp

class ForecasterFactory:
    def __init__(self, input_size=constants.detectorDefaults.win_size):
        self._input_size = input_size

        self.factory = dict()
        self.factory['test'] = HyperTest()
        self.factory['linear'] = HyperLinear()
        self.factory['gru'] = HyperGru()
        self.factory['lstm'] = HyperLstm()
        self.factory['cnn'] = HyperCNN()
        self.factory['hybrid'] = HyperHybrid()

    def available(self):
        if getattr(self, '_summary', None) is None:
            self._summary = []
            for model in self.factory.keys():
                self._summary.append(
                    dict(
                        architecture=self.factory[model].builder_id, 
                        description=self.factory[model].__repr__(),
                    )
                )
        return self._summary
    
    def description(self, architecture: str) -> str:
        if self.has(architecture):
            return self.factory[architecture].__repr__()
        return ""
    
    def has(self, architecture:str) -> bool:
        return architecture in self.factory.keys()

    @property
    def input_size(self):
        return self._input_size

    def build(self, architecture: str):
        result = None
        if self.has(architecture):
            result = self.factory[architecture].build(self.input_size)
        return result