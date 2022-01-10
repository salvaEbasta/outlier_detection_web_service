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
df = pd.DataFrame({"d": ["2014-01-02", "2015-01-02"]})
df["d"] = pd.to_datetime(df["d"])
df["p"] = [np.nan]*len(df)
df["v"] = [1.2]*len(df)
#df = df.fillna(None).astype("string")

# %%