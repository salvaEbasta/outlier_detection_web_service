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
import numpy as np
lbls = np.array([0, 0, 0, 1, 1, 1, 2], dtype=int)
np.sum(lbls == 2)

# %%
