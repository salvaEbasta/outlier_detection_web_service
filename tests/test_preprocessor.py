import numpy as np

from ml_microservice.anomaly_detection import preprocessing

def test_split_overlapping_windows():
    window_size = 5
    s = np.ones((100), dtype=float)
    train, dev, test = preprocessing.split(s, dev=True, window_size=window_size)
    print(train.shape, dev.shape, test.shape)
    assert train.shape[0] == 70
    assert dev.shape[0] == 20 + window_size
    assert test.shape[0] == 10 + window_size

def test_window_building():
    input_width = 3
    label_width = 1
    total_width = input_width + label_width
    train = np.array([0,1,2,3,4,5,6,7,8,9,10])
    preproc = preprocessing.Preprocessor(
        train, 
        input_width=input_width, 
        label_width=label_width
    )
    X, y = preproc.extract_windows(train)
    print(X)
    print(y)
    assert len(X) == len(train) - total_width + 1
    assert len(X) == len(y)
    assert X[0][0] == 0
    assert X[0][1] == 1
    assert X[0][2] == 2
    assert y[0][0] == 3
    assert X[3][0] == 3
    assert X[3][1] == 4
    assert X[3][2] == 5
    assert y[3][0] == 6

def test_window_building_small_input():
    input_width = 3
    label_width = 1
    total_width = input_width + label_width
    train = [1]
    preproc = preprocessing.Preprocessor(
        train, 
        input_width=input_width, 
        label_width=label_width
    )
    X, y = preproc.extract_windows(train)
    print(X)
    print(y)
    assert len(X) == 1
    assert len(X) == len(y)
    assert X[0][0] == X[0][1] == X[0][2] == 0
    assert y[0][0] == 1

def test_make_dataset():
    input_width = 3
    label_width = 1
    total_width = input_width + label_width
    train = np.array([0,1,2,3,4,5,6,7,8,9,10])
    preproc = preprocessing.Preprocessor(
        train, 
        input_width=input_width, 
        label_width=label_width
    )
    X, y = preproc.train
    print(X, y)
    assert X.shape[0] == y.shape[0] == len(train) - total_width + 1
    assert X.shape[1] == input_width