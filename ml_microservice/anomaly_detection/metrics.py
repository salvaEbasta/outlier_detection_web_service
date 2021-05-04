import tensorflow as tf
from tensorflow import keras

class NaiveRegressor(keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        results = inputs[..., -1]
        return tf.expand_dims(results, axis=-1)

def naive_model_metric(X, y, y_hat):
    """
        Evaluate a regressor (y_hat) against a naive model.\n
        Mse(regressor)/Mse(naive) -> good results come if result << 1\n
    """
    naive = NaiveRegressor()
    naive.compile(loss="mse")
    y_naive = naive.predict(X)
    assert y.shape == y_hat.shape == y_naive.shape
    mse = keras.losses.MeanSquaredError()
    return mse(y, y_hat).numpy() / mse(y, y_naive).numpy()