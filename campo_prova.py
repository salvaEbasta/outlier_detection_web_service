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

# %%
from xml.etree import ElementTree as ET

with open('data/xml/s5_2012_samples.xml', 'r') as f:
    xml = f.read().replace('\n', '')
# %%
import pandas as pd

dfa = pd.DataFrame({
    'a': [1,2,3,4,5],
    'b': [1,2,3,4,5]
})

dfb = pd.DataFrame({
    'a': [6,7,8],
    'c': [6,7,8]
})

dfc = dfa.append(dfb).fillna(.0)

# dfc = dfa.append(dfb)
# dfc.fillna(0.0)
# %%
from argparse import Namespace
p = Namespace()
p.prova = 1
print(p)
# %%
class Prova():
    def __init__(self, var = 1):
        self._var = var
    
    def load(self, var=2):
        self.__init__(var=var)
        print(self)
    
    def __repr__(self):
        return "Prova(var: {:f})".format(self._var)
p = Prova()
print(p._var)
p.load()
print(p._var)
# %%
import logging
class Prova():
    def __init__(self):
        logging.basicConfig(filename=os.path.join("data/log", "prova.log"), level=logging.INFO)
        logging.info("Prova log")
    
    def write(self, msg):
        logging.warning(str(msg))
p = Prova()
p.write('Prova prova')
# %%
import os
import pandas as pd

df = pd.DataFrame({'p1': [1,2,], 'p2': [3,4]})
tmp = pd.DataFrame(df)
for idx, row in df.iterrows():
    print(f"{idx}: {row.to_dict()}")
    tmp = tmp.append(row, ignore_index=True)
tmp.iloc[-1]['p1']

d1 = "2014-01-02"
d1 = pd.to_datetime(d1)
d2 = "2014-01-04"
d2 = pd.to_datetime(d2)
d1.year == d2.year and d1.month == d2.month and d1.day == d2.day

# %%
