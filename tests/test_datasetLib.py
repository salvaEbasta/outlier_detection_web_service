import os

from ml_microservice import service_logic

def test_list_datasets():
    dsets = os.listdir('data/datasets')
    dlib = service_logic.DatasetsLibrary()
    dlist = dlib.datasets
    assert len(dlist) == len(dsets)
    assert sum([doc in dlist for doc in dsets]) == len(dsets)

def test_has_dataset():
    dsets = os.listdir('data/datasets')
    dlib = service_logic.DatasetsLibrary()
    for d in dsets:
        assert dlib.has(d)

def test_fetch_dataset():
    dsets = os.listdir('data/datasets')
    dlib = service_logic.DatasetsLibrary()
    dframe = dlib.fetch(dsets[0])
    assert len(dframe)