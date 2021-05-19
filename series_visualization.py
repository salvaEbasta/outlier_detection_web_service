# %%
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
# %%
print("Serie 1")
s1 = pd.read_csv('data/datasets/s1_somma_progressiva.csv')
fig, axs = plt.subplots(len(s1.columns)-1)
for i in range(1, len(s1.columns)):
    axs[i-1].plot(s1[s1.columns[i]])
    axs[i-1].set_title(s1.columns[i])
# %%
print("Serie 2")
s1 = pd.read_csv(glob.glob('data/datasets/s2_*.csv')[-1])
fig, axs = plt.subplots(len(s1.columns)-1)
for i in range(1, len(s1.columns)):
    axs[i-1].plot(s1[s1.columns[i]])
    axs[i-1].set_title(s1.columns[i])
# %%
fig, axs = plt.subplots(2)
i = 0
for csv in glob.glob('data/datasets/s3_*'):
    print(f"{csv}")
    s3 = pd.read_csv(csv)
    print(s3.columns[10])
    axs[i].plot(s3[s3.columns[10]])
    axs[i].set_title(f"{csv}: {s3.columns[10]}")
    i += 1
# %%
print("Series 4")
s4 = pd.read_csv(glob.glob('data/datasets/s4_*')[-1])
print(s4.columns[10])
tot = 7
fig, axs = plt.subplots(tot)
for i in range(0, tot):
    axs[i].plot(s4[s4.columns[i+1]])
    axs[i].set_title(f"{s4.columns[i+1]}")
# %%
print("Serie 11")
s11 = pd.read_csv('data/datasets/s11.csv')
plt.plot(s11)
# %%
