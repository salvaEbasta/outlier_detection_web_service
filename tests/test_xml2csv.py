import os

from tests import TEST_DIR
from ml_microservice import configuration as c
from ml_microservice.conversion import Xml2Csv as x2c

XML_DIR = os.path.join(TEST_DIR, "xml")
XML_PLURI = os.path.join(XML_DIR, 's5_2012_samples.xml')
XML_MONO = os.path.join(XML_DIR, 's11_2012_samples.xml')

def read_xml(path):
    with open(path, 'r') as f:
        content = "".join(f.read().replace("\n", ""))
    return content

def test_well_formed_xml_aggregator():
    content = read_xml(XML_PLURI)
    print(content[:1000])

    converter = x2c()
    dfs = converter.parse(content)

    cols = None
    length = -1
    for dfID, df in dfs.items():
        if cols is None:
            cols = set(df.columns)
        if length < 0:
            length = len(df)
        print(dfID)
        assert length == len(df)
        assert len(cols.difference(set(df.columns))) == 0
        assert not any(df[c.timeseries.date_column].duplicated())

def test_monodimensional_aggregator():
    content = read_xml(XML_MONO)
    print(content[:1000])

    converter = x2c()
    dfs = converter.parse(content)
    
    cols = None
    length = -1
    for dfID, df in dfs.items():
        print(dfID)
        if cols is None:
            cols = set(df.columns)
        if length < 0:
            length = len(df)
        assert length == len(df)
        assert len(cols.difference(set(df.columns))) == 0
        assert not any(df[c.timeseries.date_column].duplicated())