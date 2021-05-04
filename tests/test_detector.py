import configparser
import json
import os
import shutil

import numpy as np

from ml_microservice import strings
from ml_microservice.anomaly_detection import preprocessing
from ml_microservice.anomaly_detection import detector

conf = configparser.ConfigParser()
conf.read(strings.config_file)

def test_detector_dimensions():
    window_size = 3
    train = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    test = np.array([20,21,22,23,24,25,100,27,28,29,30,100])
    
    preproc = preprocessing.Preprocessor(train=train, test=test, input_width=window_size)

    d = detector.Detector(window_size, regressor_model='test')
    d.fit(*preproc.train)
    anomalies = d.detect(*preproc.test)
    print(anomalies)
    assert len(anomalies.shape) == 2 and anomalies.shape[0] == preproc.test[0].shape[0]
    #assert np.sum(np.squeeze(anomalies)) == 2

def test_save_n_load():
    #w_size = 99
    #l = 0.3
    #k = 3
    label = 'detector_save_test'
    path = os.path.join(conf['Series']['path'], label)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    d = detector.Detector()
    old_ws = d.window_size
    old_l = d._lambda
    old_k = d._k
    old_reg_mod = d._regressor_model
    d.save(path)
    assert strings.detector_param_file in os.listdir(path)    
    with open(os.path.join(path, strings.detector_param_file), 'r') as f:
        params: dict = json.load(f)
    
    assert int(params.get('window_size', None)) == old_ws
    assert float(params.get('l', None)) == old_l
    assert float(params.get('k', None)) == old_k
    assert params.get('regressor_model', None) == old_reg_mod
    
    d = detector.Detector(path=path)

    assert d.window_size == old_ws
    assert d._lambda == old_l
    assert d._k == old_k
    assert d._regressor_model == old_reg_mod

    shutil.rmtree(path)