import tensorflow as tf
from tensorflow import keras

MODEL_BUILDER = dict()

def test_model():
    test = keras.Sequential([
        keras.layers.Dense(1, activation='relu', input_shape=(10,)),
    ])
    test.compile(loss='mse')
    return test
MODEL_BUILDER['test'] = test_model