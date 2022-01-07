import os
import shutil
import json
import re

import pytest

import numpy as np

from ml_microservice import constants as c
from ml_microservice.anomaly_detection import preprocessing
from ml_microservice.service_logic import DetectorTrainer
from ml_microservice.service_logic import TimeseriesLibrary

def _detector_init(mid, training, forecasting):
    dl = DetectorTrainer()
    if os.path.exists(os.path.join(dl.storage, mid)):
        shutil.rmtree(os.path.join(dl.storage, mid))
    result = dl.train(mid, training, forecasting)
    return result

def test_list():
    mid = 'test_list'
    training = dict(label='jena', dataset='temperature', column='T (degC)')
    forecaster = 'test'

    dlib = DetectorTrainer()
    models = dlib.detectors_list
    versions = [v for m in models for v in m['versions'] if m['model'] == mid]
    
    tmp = dlib.train(mid, training, forecaster)
    assert tmp['id'] == mid
    assert tmp['training']['column'] == training['column'] and \
        tmp['training']['dataset'] == training['dataset'] and \
        tmp['training']['label'] == training['label']
    models = dlib.detectors_list
    assert type(models) == list
    assert all([c.formats.version % 0 in f['versions'] for f in models])
    assert mid in [m['model'] for m in models]
    assert len([v for m in models for v in m['versions'] if m['model'] == mid]) == len(versions) + 1
    path = os.path.join(dlib.storage, mid)
    shutil.rmtree(path)

def test_train():
    mid = 'test_train'
    training = dict(label='jena', dataset='temperature', column='T (degC)')
    forecaster = 'test'

    result = _detector_init(mid, training, forecaster)
    print(result)
    assert result.get('id', None) == mid
    assert result.get('version', None) == c.formats.version % 0
    assert result.get('training', None) is not None
    assert result['training'].get('label', None) == training['label']
    assert result['training'].get('dataset', None) == training['dataset']
    assert result['training'].get('column', None) == training['column']
    assert result.get('forecasting_model', None) == forecaster
    assert result.get('training_performance', None) is not None
    p = result['training_performance']
    assert p.get('naive_score', -1) > 0
    assert len(p.get('y', None)) > 0
    assert len(p.get('anomalies', None)) > 0
    assert len(p['anomalies']) == len(p['y'])
    assert p.get('total_time', -1) > 0

    dt = DetectorTrainer()
    path = os.path.join(dt.storage, mid)
    assert os.path.exists(path)
    version_path = os.path.join(path, c.formats.version%0)
    assert os.path.exists(version_path)
    summary_file = os.path.join(version_path, c.files.detector_summary)
    assert os.path.exists(summary_file)

    with open(summary_file, 'r') as f:
        summ = json.load(f)
    print(summ)
    assert summ.get('status', None) is not None
    assert summ.get('created_on', None) is not None
    assert summ.get('training', None) is not None
    assert summ['training'].get('label', None) == training['label']
    assert summ['training'].get('dataset', None) == training['dataset']
    assert summ['training'].get('column', None) == training['column']
    shutil.rmtree(path)

def test_prediction():
    # Init dataset
    samples = 1000

    dl = TimeseriesLibrary()
    data, column = dl.fetch_ts('jena', 'temperature')

    # Init detector
    mid = 'test_prediction'
    training = dict(label='jena', dataset='temperature', column='T (degC)')
    forecaster = 'test'
    assert column == training['column']

    _detector_init(mid, training, forecaster)
    dt = DetectorTrainer()
    v, env = dt.load_detector(mid)
    assert v == c.formats.version%0
    assert dt.loaded

    result = dt.detect(data[-samples:])
    print(result)
    assert result.get('total_time', None) > 0
    assert len(result.get('anomalies', None)) > 0
    assert result.get('start_detection_idx', None) == dt._detector.window_size
    assert len(result.get('anomalies')) + dt._detector.window_size == len(result.get('data', None))
    anomalies = np.array(result['anomalies'])
    print(anomalies.shape)
    assert len(anomalies.shape) == 2
    assert anomalies.shape[1] == 1

    assert len(result.get('rmse', None)) == len(result.get('naive_score', None)) == len(result.get('anomalies'))
    assert result.get('degradation', None) == 'undetected'
    shutil.rmtree(os.path.join(dt.storage, mid))
