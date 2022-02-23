from ml_microservice.anomaly_detection.factory import Factory

f = Factory()
models = [
    "WindowedGaussian",
    "DeepAnT",
    "GRU",
    "LSTM",
    "SARIMAX",
    "Prophet",
]

def test_has():
    has_not = 'prova'
    assert not f.has(has_not)
    assert all(map(lambda x: f.has(x), models))

def test_tuners():
    has_not = 'prova'
    assert f.get_tuner(has_not) is None
    assert all(map(lambda x: f.get_tuner(x) is not None, models))

def test_loaders():
    has_not = 'prova'
    assert f.get_loader(has_not) is None
    assert all(map(lambda x: f.get_loader(x) is not None, models))
