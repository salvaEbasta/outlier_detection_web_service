import numpy as np
from tensorflow import keras
import time
import os
import json
from configparser import ConfigParser

conf = ConfigParser()
conf.read('config.ini')


def test():
    return {'test': 'test012'}, 200


def models_list():
    #return {'models': [f for f in os.listdir(conf['Resources']['modelsPath'])]}, 200
    summary = os.path.join(
        conf['Resources']['modelsPath'],
        conf['Resources']['modelsSummary']
    )
    with open(summary, 'r') as f:
        resp = json.load(f)
    return resp, 200

def models_train_new(payload: dict):
    return {'error': 'Not implemented'}, 502


def model_info(model:str):
    modelsDir = conf['Resources']['modelsPath']
    if model not in [d for d in os.listdir(modelsDir) if os.path.isdir(os.path.join(modelsDir, d))]:
        return {'error': 'Model %s not found' % model}, 404
    modelPath = os.path.join(modelsDir, model)
    model = keras.models.load_model(modelPath)
    info = []
    model.summary(print_fn=lambda x: info.append(x))
    return {'info': f"{info}"}

def model_predict(model: str, payload: dict):
    modelsDir = conf['Resources']['modelsPath']
    if model not in [d for d in os.listdir(modelsDir) if os.path.isdir(os.path.join(modelsDir, d))]:
        return {'error': 'Model %s not found' % model}, 404
    modelPath = os.path.join(modelsDir, model)
    model = keras.models.load_model(modelPath)
    print(f"[.] Received payload({type(payload)}): {payload}")
    X = np.array(payload['data'])
    print(f"[.] Data shape: {X.shape}")
    tStart = time.time()
    y = model.predict(X)
    tDelta = time.time() - tStart
    print("[.] Prediction: {} ({:.8f}s)".format(y, tDelta))
    return {"time": "{:.8f}".format(tDelta),
                "prediction": "{}".format(y)
            }, 200

def model_retrain(model: str, payload: dict):
    return {'error': 'Not implemented'}, 502
