import os
import shutil
import configparser
import json

import pytest

from ml_microservice.service_logic import DatasetsLibrary, DetectorsLibrary
import ml_microservice.strings as strs

conf = configparser.ConfigParser()
conf.read(strs.config_file)

def test_list():
    detectLib = DetectorsLibrary()
    series = detectLib.list
    assert type(series) == list
    found = False
    label = 'test'
    version = strs.version_format % 0
    for s in series:
        if s["serie"] == label and version in s["versions"]:
            found = True
            break
    assert found

def test_assembly():
    label = 'test'
    blueprint = dict(
        label=label, 
        regressor='test', 
        dataset='jena_T_test.csv',
    )
    
    path = os.path.join(conf['Series']['path'], label)
    if os.path.exists(path):
        shutil.rmtree(path)

    dl = DetectorsLibrary()
    result = dl.assemble(blueprint)
    print(result)
    assert result.get('status', None) is not None
    assert result.get('created_on', None) is not None
    assert result.get('training_time', None) is not None
    assert result.get('regressor', None) == blueprint['regressor']
    assert result.get('dataset', None) == blueprint['dataset']
    assert result.get('column', None) is not None
    assert result.get('regression_performance', None) is not None
    assert 'epochs' in result and result['epochs'] >= 1

    assert os.path.exists(path)
    version_path = os.path.join(path, strs.version_format% 0 )
    assert os.path.exists(version_path)
    summary_file = os.path.join(version_path, strs.model_summary_file)
    assert os.path.exists(summary_file)