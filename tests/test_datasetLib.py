import os
import re
import shutil

import pandas as pd

from ml_microservice import configuration as c
from ml_microservice import service_logic
from ml_microservice.conversion import Xml2Csv
from tests import TEST_DIR
from tests.test_xml2csv import XML_PLURI
from tests.test_xml2csv import read_xml

TS_DATA = os.path.join(TEST_DIR, "timeseries")
GROUP = "testTSLib"

def _init(group):
    if os.path.exists(TS_DATA):
        shutil.rmtree(TS_DATA)
    os.makedirs(TS_DATA)
    ts_lib = service_logic.TimeseriesLibrary(path = TS_DATA)
    x2c = Xml2Csv()
    dfs = x2c.convert(read_xml(XML_PLURI))
    for dfID, df in dfs.keys():
        ts_lib.save(GROUP, dfID, df)

def _close():
    shutil.rmtree(TS_DATA)

def test_list_datasets():
    _init(GROUP)

    ts_list = os.listdir(TS_DATA)
    ts_lib = service_logic.TimeseriesLibrary(path = TS_DATA)
    tss = ts_lib.timeseries
    print(tss)
    assert len(tss) == len(ts_list)
    
    _close()

def test_has_dataset():
    group = GROUP
    _init(group)
    dim = "Classifica_su_venduto"

    ts_lib = service_logic.TimeseriesLibrary(path = TS_DATA)
    assert ts_lib.has(group, dim)
    assert ts_lib.has_group(group)
    assert ts_lib.has_dimension(dim)

    _close()

def test_fetch_dataset():
    _init()

    group = GROUP
    dim = "Classifica_su_venduto"
    
    x2c = Xml2Csv()
    dfs = x2c.convert(read_xml(XML_PLURI))
    assert dim in dfs.keys()
    df = dfs[dim]

    ts_lib = service_logic.TimeseriesLibrary()
    fetched = ts_lib.fetch(group, dim)

    assert fetched is not None
    assert len(fetched) == len(df)
    assert not any(fetched[c.timeseries.date_column].duplicated())
    assert len(set(df.columns).difference(set(fetched.columns)) ) == 0

    _close()

def test_save():
    group = GROUP + "_test_save"
    
    x2c = Xml2Csv()
    dfs = x2c.convert(read_xml(XML_PLURI))

    ts_lib = service_logic.TimeseriesLibrary(path = TS_DATA)    

    for dfID, df in dfs.items():
        ts_lib.save(group, dfID, df)

    group_path = os.path.join(TS_DATA, group)
    assert os.path.exists(group_path)
    assert len(dfs) == os.listdir(group_path)
    for dfID in dfs.keys():
        assert any([
            re.match(".+{:s}.csv".format(dfID), f) is not None 
                for f in os.listdir(group_path)
        ])
        saved = pd.read_csv(os.path.join(group_path, "{:s}.csv".format(dfID)))
        assert len(saved) == len(df)
        assert len(set(saved.columns).difference(set(df.columns))) == 0
        
    shutil.rmtree(group_path)