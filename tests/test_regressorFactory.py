from ml_microservice.anomaly_detection import model_factory

def test_has():
    label = 'test'
    mf = model_factory.RegressorFactory()
    assert mf.has(label)

def test_architecture():
    label = 'test'
    mf = model_factory.RegressorFactory()
    assert mf.has(label)
    #print(mf.architecture(label))
    assert mf.architecture(label) is not None

def test_available():
    label = 'test'
    mf = model_factory.RegressorFactory()
    available = mf.available()
    assert len(available) > 0
    found = False
    for d in available:
        if d['name'] == label and d['architecture'] is not None:
            found = True
            break
    assert found

def test_build():
    label = 'test'
    mf = model_factory.RegressorFactory()
    assert mf.has(label)
    model = mf.build(label)
    assert model is not None
    #print(model.to_json())
    #assert len(model.to_json()) == len(mf.architecture(label))