import configparser
import json
import os
import shutil

import numpy as np

from ml_microservice import constants as c
from ml_microservice.anomaly_detection import preprocessing
from ml_microservice.anomaly_detection import detector

def test_detector_dimensions():
    window_size = 3
    train = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    dev = np.array([0,1,2,3,4,5,6,7,8])
    test = np.array([20,21,22,23,24,25,100,27,28,29,30,100])
    
    preproc = preprocessing.Preprocessor(train=train, dev=dev, test=test, input_width=window_size)

    d = detector.Detector(window_size=window_size, forecasting_model='test')
    d.fit(*preproc.train, dev_data=preproc.dev)
    anomalies, y_hat, history = d.detect(*preproc.test)
    print(anomalies)
    assert len(anomalies.shape) == 2 and anomalies.shape[0] == preproc.test[0].shape[0]
    assert len(anomalies) == len(y_hat)
    assert len(history.rmse) == len(y_hat)

def test_save_n_load():
    window_size = 3
    train = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    dev = np.array([0,1,2,3,4,5,6,7,8,9,10])
    #w_size = 99
    #l = 0.3
    #k = 3
    label = 'test_detector_save'
    path = os.path.join(c.detectorTrainer.path, label)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    d0 = detector.Detector(window_size=window_size, forecasting_model='test')
    pp = preprocessing.Preprocessor(train=train, dev=dev, input_width=window_size)
    d0.fit(*pp.train, dev_data=pp.dev)

    old_ws = d0.window_size
    old_l = d0._lambda
    old_k = d0._k
    old_reg_mod = d0._forecasting_model
    d0.save(path)
    assert c.files.detector_params in os.listdir(path)
    assert c.files.detector_history in os.listdir(path)    
    with open(os.path.join(path, c.files.detector_params), 'r') as f:
        params: dict = json.load(f)
    
    assert params.get('window_size', None) == old_ws
    assert params.get('l', None) == old_l
    assert params.get('k', None) == old_k
    assert params.get('forecasting_model', None) == old_reg_mod
    
    d1 = detector.Detector(path=path)
    print(d0.params)
    print(d1.params)
    assert d0.params == d1.params
    shutil.rmtree(path)

def test_history_update():
    y, y_hat, y_naive = [0 ,1, 2], [0, .5, 1], [0, 0, 1]
    h = detector.History()
    v = h.values
    assert 'y' in v
    assert 'y_hat' in v
    assert 'y_naive' in v
    assert 'timestamp' in v
    assert 'datapoints' in v
    assert 'mse' in v
    assert 'mse_naive' in v
    h.update_state(y, y_hat, y_naive)

    assert len(h.values['y']) == 3 and len(h.values['y_hat']) == 3 and len(h.values['y_naive']) == 3
    assert len(h.rmse) == 3
    assert len(h.naive_score) == 3
