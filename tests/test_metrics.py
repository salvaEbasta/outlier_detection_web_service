import numpy as np

from ml_microservice.anomaly_detection import preprocessing
from ml_microservice.anomaly_detection import metrics

def test_naive_model():
    window_size = 3
    train = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    test = np.array([20,21,22,23,24,25,100,27,28,29,30,100])
    
    preproc = preprocessing.Preprocessor(train=train, test=test, input_width=window_size)

    naive = metrics.NaivePredictor()
    naive.compile(loss="mse")
    #naive.fit(*preproc.train)
    X, y = preproc.extract_windows(test)
    y_naive = naive.predict(X)
    print(y_naive)
    assert len(y_naive.shape) == 2 and y_naive.shape[0] == preproc.test[0].shape[0] and y_naive.shape[1] == 1
    assert np.sum(y_naive - X[...,-1]) == 0

def test_naive_metric():
    window_size = 3
    train = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    test = np.array([20,21,22,23,24,25,100,27,28,29,30,100])
    
    preproc = preprocessing.Preprocessor(train=train, test=test, input_width=window_size)

    naive = metrics.NaivePredictor()
    naive.compile(loss="mse")
    #naive.fit(*preproc.train)
    X_test, y_test = preproc.test
    y_naive = naive.predict(X_test)
    metric = metrics.naive_model_metric(X_test, y_test, y_naive)
    assert metric == 1
