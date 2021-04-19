import tensorflow as tf
from tensorflow import keras

class Baseline(keras.Model):
    def __init__(self):
        super().__init__()
    
    def call(self, inputs):
        results = inputs[..., -1]
        return results[..., tf.newaxis]

def naive_model_metric(X, y, predictions):
    naive_prediction = X[..., -1]
    naive_error = keras.metrics.MeanSquaredError()
    naive_error.update_state(y, naive_prediction)

    model_error = keras.metrics.MeanSquaredError()
    model_error.update_state(y, predictions)
    return model_error.result().numpy()/naive_error.result().numpy()