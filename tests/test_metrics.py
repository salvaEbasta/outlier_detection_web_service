import numpy as np

from ml_microservice.anomaly_detection import metrics

def test_naive_prediction():
    ts = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    y_hat = metrics.naive_prediction(ts)
    assert len(y_hat) == len(ts)
    assert y_hat[0] == ts[0]
    assert all(map(
        lambda t: t[0] - t[1] < np.finfo(float).eps, 
        zip(y_hat[1:], ts[:-1])
    ))

def test_naive_metric():
    ts = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    y_naive = metrics.naive_prediction(ts)

    assert metrics.naive_metric(ts, np.zeros(ts.shape), y_naive) > 1
    assert metrics.naive_metric(ts, ts, y_naive) < 1
