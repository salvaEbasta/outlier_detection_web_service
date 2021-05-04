# %%
def prova_1():
    import os
    print(os.listdir())
    import pandas as pd
    import numpy as np

    jena_frame = pd.read_csv('../data/datasets/jena_climate_2009_2016.csv')
    jena_T_frame = pd.read_csv('../data/datasets/jena_T_test.csv')
# %%
def unpacking_prova(x, y, epochs=1):
    print("x: {}, y: {}, epochs: {}".format(x,y,epochs))

t = ([0,0,0], [1])
epochs = 20
unpacking_prova(*t)
unpacking_prova(*t, epochs=epochs)
# %%
