import os
import shutil

import pandas as pd

from ml_microservice import constants as c
from ml_microservice.conversion import Xml2Csv as x2c

def test_well_formed_xml_aggregator():
    xml = 's5_2012_samples.xml'
    # Read xml
    xml_path = os.path.join(c.xml.path, xml)
    with open(xml_path, 'r') as f:
        xml_str = ''.join(f.read().replace('\n', ''))
    print(xml_str[:1000])

    converter = x2c()
    assert not converter.is_mono(xml_str)
    dsets = converter.parse(xml_str)

    cols = set(dsets[0][1].keys())
    length = len(dsets[0][1][next(iter(dsets[0][1].keys()))])
    for name, data in dsets:
        print(name)
        assert len(cols.difference(set(data.keys()))) == 0
        assert len(set(data.keys()).difference(cols)) == 0
        for col in data.keys():
            assert length == len(data[col])

def test_monodimensional_aggregator():
    xml = 's11_2012_samples.xml'
    # Read xml
    xml_path = os.path.join(c.xml.path, xml)
    with open(xml_path, 'r') as f:
        xml_str = ''.join(f.read().replace('\n', ''))
    print(xml_str[:1000])

    converter = x2c()
    assert converter.is_mono(xml_str)
    dsets = converter.parse(xml_str)

    cols = set(dsets[0][1].keys())
    length = len(dsets[0][1][next(iter(dsets[0][1].keys()))])
    for name, data in dsets:
        print(name)
        print(data)
        assert len(cols.difference(set(data.keys()))) == 0
        assert len(set(data.keys()).difference(cols)) == 0
        for col in data.keys():
            assert length == len(data[col])