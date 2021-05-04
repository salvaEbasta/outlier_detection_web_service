import configparser
import os
import shutil

import numpy as np

from ml_microservice import service_logic
from ml_microservice import strings

conf = configparser.ConfigParser()
conf.read(strings.config_file)

def _detector_init(label):
    """
        Fresh new detector @/<-storage->/<-label->/v0/\n
        Regressor model: test
    """
    storage = conf['Series']['path']
    serie_dir = os.path.join(storage, label)
    if os.path.exists(serie_dir):
        shutil.rmtree(serie_dir)
    
    blueprint = dict(
        label=label,
        regressor='test',
        dataset='jena_T_test.csv'
    )

    dl = service_logic.DetectorsLibrary()
    dl.assemble(blueprint)
    return blueprint


def test_info():
    label = 'test'
    _detector_init(label)

    ad = service_logic.AnomalyDetection(label=label)
    info = ad.info
    assert info.get('status', None) is not None
    assert info.get('created_on', None) is not None
    assert info.get('regression_performance', None) is not None
    assert info.get('training_time', None) is not None
    assert info.get('dataset', None) is not None
    assert info.get('column', None) is not None
    assert info.get('regressor', None) is not None
    assert info.get('l', None) is not None
    assert info.get('window_size', None) is not None
    assert info.get('k', None) is not None
    assert info.get('variance', None) is not None

def test_prediction():
    # Init dataset
    samples = 1000

    dl = service_logic.DatasetsLibrary()
    dframe = dl.fetch('jena_T_test.csv')
    dframe = dframe[dframe.columns[1]]
    
    data = dframe[-samples:]

    # Init detector
    label = 'test'
    _detector_init(label)

    ad = service_logic.AnomalyDetection(label=label)
    result = ad.predict(data)
    assert result.get('total_time', None) is not None
    assert result.get('anomalies') is not None
    assert result.get('not_evaluated_until', None) is not None

    assert result['not_evaluated_until'] == ad._detector.window_size - 1

    anomalies = np.array(result['anomalies'])
    print(anomalies.shape)
    assert len(anomalies.shape) == 2
    assert anomalies.shape[0] == samples - ad._detector.window_size
    assert anomalies.shape[1] == 1

def test_update():
    # Init detector
    label = 'test'
    _detector_init(label)

    # Init data
    samples = 1000
    
    dl = service_logic.DatasetsLibrary()
    dframe = dl.fetch('jena_T_test.csv')
    dframe = dframe[dframe.columns[1]]
    
    data = dframe[-samples:]
    epochs = 2

    ad = service_logic.AnomalyDetection(label=label)
    assert ad.detector_ready
    result = ad.update(data, epochs=epochs)
    assert 'training_time' in result and result['training_time'] > 0
    assert 'new_data_points' in result and result['training_time'] > 0
    assert 'regression_performance' in result and result['regression_performance'] > 0
    assert 'epochs' in result and result['epochs'] == 2