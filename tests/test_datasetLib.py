import os
import shutil

import pandas as pd

from ml_microservice import constants as c
from ml_microservice import service_logic
from ml_microservice.conversion import Xml2Csv as x2c

def test_list_datasets():
    dsets = os.listdir('data/datasets')
    dlib = service_logic.DatasetsLibrary()
    dlist = dlib.datasets
    print(dlist)
    assert len(dlist) == len(dsets)

def test_has_dataset():
    label = 'jena'
    dataset = 'original'
    dlib = service_logic.DatasetsLibrary()
    assert dlib.has(label, dataset)

def test_fetch_dataset():
    label = 'jena'
    dataset = 'original'
    dlib = service_logic.DatasetsLibrary()
    dframe = dlib.fetch(label, dataset)
    assert len(dframe)

def test_save():
    label = 'test'
    xml = 's11_2012_samples.xml'
    xml_path = os.path.join(c.xml.path, xml)
    with open(xml_path, 'r') as f:
        xml_str = ''.join(f.read().replace('\n', ''))
    converter = x2c()
    dsets = converter.parse(xml_str)
    
    dlib = service_logic.DatasetsLibrary()
    for name, data in dsets:
        dlib.save(label, name, data)

    dest_path = os.path.join(dlib.storage, label)
    assert os.path.exists(dest_path) and len(os.listdir(dest_path)) == 1
    print(os.listdir(dest_path))

    dframes = [(d[:-4], pd.read_csv(os.path.join(dest_path, d), index_col='Unnamed: 0')) for d in os.listdir(dest_path)]
    for k, v in dframes:
        match = False
        for n, d in dsets:
            if k == n:
                match = True
                break
        assert match
        assert len(set(v.columns).difference(set(d.keys()))) == 0
        assert len(set(d.keys()).difference(set(v.columns))) == 0
        for col in d.keys():
            assert len(v[col]) == len(d[col])
    shutil.rmtree(dest_path)