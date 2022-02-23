# %%
"""
    Setup microservice
"""
import os
import shutil
import sys
from requests import get, post

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wdir = os.path.abspath(os.curdir)
if "ml_microservice" not in os.listdir(wdir):
    wdir = os.path.abspath(os.path.join(wdir, ".."))
    os.chdir(wdir)
    sys.path.append(wdir)

from ml_microservice import configuration as c

wdir = os.path.abspath(os.curdir)
demo_data_folder = os.path.join(wdir, os.path.join("scripts", "demo_data"))

URL = "http://localhost:5000/api/"
print(f"Service available @{URL}")













# %%
"""
    Load XML -> timeseries initialization
"""
print("------------------XML conversion and timeseries initialization---------")
groupID = 'demo'
if groupID in os.listdir(c.timeseries.path):
    shutil.rmtree(os.path.join(c.timeseries.path, groupID))

current_xml = 'demo_s11_2017_samples.xml'

with open(os.path.join(demo_data_folder, current_xml), 'r') as f:
    xml_body = ''.join(f.read().replace('\n', ''))

r = get(URL + "timeseries").json()
print(f"[{r['code']}] Available groups: {r['timeseries']}")
























# %%
r = post(URL + "convert/xml", json = dict(
    xml = xml_body,
    store = dict(
        group = groupID,
        override = True,
    ),
)).json()
print(f"[{r['code']}] Xml conversion: {r}")


























# %%
r = get(URL+"timeseries").json()
print("Available timeseries, groups: ", [d['group'] for d in r['timeseries']])
print("Found group \'{}\': {}".format(
    groupID, 
    groupID in [d['group'] for d in r['timeseries']]
))
dims = [d['dimensions'] for d in r['timeseries']
        if d['group'] == groupID][0]
print(f"Group \'{groupID}\': dimensions - {dims}")

dimID = dims[0]
r = get(URL + f"timeseries/{groupID}/{dimID}").json()
print(f'Group \'{groupID}\'-\'{dimID}\': shape - {r["group"]["dimension"]["shape"]}')

tsID = r['group']['dimension']['tsIDs'][0]
print(f"Group \'{groupID}\'-\'{dimID}\': timeseries - {r['group']['dimension']['tsIDs']}")
r = get(URL + f"timeseries/{groupID}/{dimID}/{tsID}").json()
ts = pd.Series(r["values"], index = list(r["values"].keys()))
ts.plot()



















# %%
# Dump XML -> dataset growth
print("-------------Timseries growth---------------------------")
current_xml = 'demo_s11_2018_samples.xml'
with open(os.path.join(demo_data_folder, current_xml), 'r') as f:
    xml_body = ''.join(f.read().replace('\n', ''))

# Only conversion, no persistence
r = post(URL + "convert/xml", json = dict(
    xml = xml_body,
)).json()
print("Xml conversion: {}".format(r))

# Same shape
r = get(URL + f"timeseries/{groupID}/{dimID}").json()
print(f'Group \'{groupID}\'-\'{dimID}\': shape - {r["group"]["dimension"]["shape"]} - unchanged')
last_value = r["group"]["dimension"]["shape"][0]
















# %%
# Persistence
r = post(URL + "convert/xml", json = dict(
    xml = xml_body,
    store = dict(
        group = groupID,
        override = False,
    ),
)).json()
print("Xml conversion: {}".format(r))
























# %%
r = get(URL + f"timeseries/{groupID}/{dimID}").json()
print(f'Group \'{groupID}\'-\'{dimID}\': shape - {r["group"]["dimension"]["shape"]}')
r = get(URL + f"timeseries/{groupID}/{dimID}/{tsID}").json()
ts = pd.Series(r["values"], index = list(r["values"].keys()))
ts.plot()
# last_value = 53
plt.vlines(last_value, max(ts), min(ts), 'r')




























# %%
# Detector creation
print("---------------Train Anomaly Detector----------------------")
mid = 'demo'
if mid in os.listdir(c.detectorTrainer.path):
    shutil.rmtree(os.path.join(c.detectorTrainer.path, mid))
method = "WindowedGaussian"

r = get(URL + "anomaly_detection").json()
print(f"Saved detectors: {r['saved']}")

r = get(URL + "anomaly_detection/methods").json()
print(f"Available anomaly detection methods: {r['methods']}")

r = post(URL + "anomaly_detection", json = dict(
    train = dict(
        groupID = groupID,
        dimID = dimID,
        tsID = tsID,
    ),
    mID = mid,
    method = method,
)).json()
print(r)

version = r["model"]["version"]
















# %%
r = get(URL + "anomaly_detection").json()
print("Saved detectors: {}".format(r))

r = get(URL + f"anomaly_detection/{mid}/{version}").json()
print(f"{mid} - {version}: Metadata: {r['metadata']}")

r = get(URL + f"anomaly_detection/{mid}/{version}/parameters").json()
print(f"{mid} - {version}: Parameters: {r['params']}")

r = get(URL + f"anomaly_detection/{mid}/{version}/history").json()
print(f"{mid} - {version}: History: {r['history']}")


























# %%
print("---------------Anomaly detection------------------")
current_xml = 'demo_s11_2019_samples.xml'
with open(os.path.join(demo_data_folder, current_xml), 'r') as f:
    xml_body = ''.join(f.read().replace('\n', ''))

r = post(URL + "convert/xml", json = dict(
    xml = xml_body,
)).json()
print("Xml conversion: {}".format(r['extracted']))

import numpy as np
for i, v in enumerate(r["extracted"][0]["data"]["net_amount"]):
    if v == "null":
        r["extracted"][0]["data"]["net_amount"][i] = np.nan

eval_ts = pd.DataFrame()
eval_ts["value"] = r["extracted"][0]["data"]["net_amount"]
eval_ts["timestamp"] = pd.to_datetime(r["extracted"][0]["data"]["timestamp"])
eval_ts["value"].plot()















# %%
r = post(URL + f"anomaly_detection/{mid}/{version}", json = dict(
    data = dict(
        values = eval_ts["value"].fillna("null").to_list(),
        dates = eval_ts["timestamp"].astype("string").to_list(),
    )
)).json()
print(f"Predictions: {r['predictions']}")
eval_ts["outlier_score"] = r["predictions"]["anomaly_score"]




























# %%
eval_ts["value"].plot()
for i, score in enumerate(eval_ts["outlier_score"]):
    if score > .95:
        plt.plot(i, eval_ts["value"][i], 'ro')


# %%
# Remove demo resources
if groupID in os.listdir(c.timeseries.path):
    shutil.rmtree(os.path.join(c.timeseries.path, groupID))

if mid in os.listdir(c.detectorTrainer.path):
    shutil.rmtree(os.path.join(c.detectorTrainer.path, mid))