# %%
# Verifica corretto funzionamento Naive Regressor
import os

import pandas as pd
import numpy as np

jena_T = pd.read_csv('data/datasets/jena_T_test.csv')
s = jena_T[jena_T.columns[1]].head(1000)

from ml_microservice.anomaly_detection import preprocessing
pp = preprocessing.Preprocessor(s)

from ml_microservice.anomaly_detection import metrics
naive = metrics.NaivePredictor()
naive.compile(loss="mse")
X, y = pp.train
y_hat = naive.predict(X)

import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.squeeze(y)[:100], label='y')
plt.plot(np.squeeze(y_hat)[:100], label='y_hat')
plt.show()
