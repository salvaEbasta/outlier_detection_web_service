import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

print(os.getcwd())

test = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1),
])
test.compile(loss='mse')
x = np.array([[0,1,2,3,4,5,6,7,8,9]])
y = np.array([[10]])
test.fit(x,y)
test.save('data/models/test')
print(test.summary())

test1 = keras.models.load_model('data/models/test')
print(test1.summary())

print(test1.predict(x))