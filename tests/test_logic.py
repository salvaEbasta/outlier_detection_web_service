import os
import shutil
import configparser
import json

import pytest

from ml_microservice.service_logic import DatasetsLibrary, Toolshed
import ml_microservice.strings as strs

conf = configparser.ConfigParser()
conf.read(strs.config_file)

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_local_datasets():
    assert os.listdir(conf['Datasets']['path'])[0] in DatasetsLibrary().datasets

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_assembly():
    label = 'test'
    model = 'test'
    path = os.path.join(conf['Series']['path'], label)
    if os.path.exists(path):
        shutil.rmtree(path)
    
    blueprint = dict(label=label, model=model, 
                        datasets=[os.listdir(conf['Datasets']['path'])[0]]
                    )
    ts = Toolshed()
    ts.assemble(blueprint)
    assert os.path.exists(path)
    version_path = os.path.join(path, strs.version_format%0)
    assert os.path.exists(version_path)
    summary_file = os.path.join(version_path, strs.model_summary_file)
    assert os.path.exists(summary_file)
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    assert summary.get('status', None) != None
    assert summary.get('architecture', 0) != 0
    assert summary.get('created_on', None) != None
    assert summary.get('model', None) == model