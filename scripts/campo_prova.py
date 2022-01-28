# %%
import os
import pandas as pd
import numpy as np

df = pd.DataFrame({'p1': [np.nan,2, np.nan, 1, 3, 3], 'p2': [np.nan, 3 ,4, np.nan, 4, 4]})
print(df["p1"].isnull().sum())
df["p2"].interpolate(method = "polynomial", order = 3)

# %%
import numpy as np
X = np.random.uniform(size=[4,3])
print(X)
centers = np.ones(shape=[2,3])
for i, v in enumerate(centers):
    print(X - v)
    print(np.linalg.norm(X - v, axis=1))
    print(np.argmin(np.linalg.norm(X - v, axis=1)))
    print(X[np.argmin(np.linalg.norm(X - v, axis=1))])

# %%
f1 = "{:s}.f1"
f2 = f1.format("{:s}.f2")
import re
re.match(f2.format(".+"), "prova.f2.f1")

# %%
import numpy as np
import pandas as pd
df = pd.DataFrame({'p1': [np.nan, 2, np.nan, 1, 3, 3], 'p2': [np.nan, 3 ,4, np.nan, 4, 4]})
df.astype("string").to_dict()
# %%
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
x = np.linspace(start= -10, stop=10, num=1000)
y = np.sin(x)
plt.plot(x, y)
plt.vlines(x[0], -1, 1, 'r')
plt.vlines(x[333], -1, 1, 'r')
plt.vlines(x[666], -1, 1, 'r')
plt.vlines(x[999], -1, 1, 'r')
fs, P = signal.periodogram(y)
p = int(1 / (fs[np.argmax(P)] + np.finfo(float).eps))
p, fs[np.argmax(P)]

# %%
class Prova():
    def __init__(self, prova=2):
        self.prova = prova
        print("prova")
    def p(self):
        print(self.prova)
    def pp(self):
        print("UwU")
p = Prova()
# %%
import os, sys
import numpy as np
import pandas as pd
ts = pd.DataFrame({
    "timestamp": [
        "2014-01-02", 
        "2014-01-03", 
        "2014-01-04", 
        "2014-01-05", 
        "2014-01-07", 
        "2014-01-08", 
        "2014-01-09", 
        "2014-01-10",
        "2014-01-11",
        ],
    "value": [
        25,
        9,
        4,
        1,
        0,
        1,
        4,
        9,
        25,
    ]
})
ts["timestamp"] = pd.to_datetime(ts["timestamp"])

ml_path = ".."
if ml_path not in sys.path:
    sys.path.append(ml_path)
from ml_microservice.anomaly_detection.models.sarimax import SARIMAX, get_exogenous

exo = get_exogenous(pd.DatetimeIndex(ts["timestamp"]))
exo

# %%
