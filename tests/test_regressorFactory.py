from ml_microservice.anomaly_detection import model_factory

def test_has():
    label = 'test'
    mf = model_factory.ForecasterFactory()
    assert mf.has(label)

def test_architecture():
    label = 'test'
    mf = model_factory.ForecasterFactory()
    assert mf.has(label)
    #print(mf.architecture(label))
    assert mf.description(label) is not None

def test_available():
    label = 'test'
    mf = model_factory.ForecasterFactory()
    available = mf.available()
    assert len(available) > 0
    found = False
    for d in available:
        if d['architecture'] == label and d['description'] is not None:
            found = True
            break
    assert found

def test_build():
    label = 'test'
    mf = model_factory.ForecasterFactory()
    assert mf.has(label)
    model = mf.build(label)
    assert model is not None
    #print(model.to_json())
    #assert len(model.to_json()) == len(mf.architecture(label))