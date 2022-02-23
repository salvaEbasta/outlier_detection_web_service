import numpy as np
import pandas as pd

from ml_microservice.anomaly_detection.transformers import Preprocessor

def test_split():
    ratio = .7
    ts_len = 100
    ts = pd.DataFrame()
    ts["value"] = np.ones([ts_len, ])
    
    pre0 = Preprocessor(ts, value_col = "value")
    ts_train, ts_test = pre0.train_test
    pre1 = Preprocessor(ts_train, value_col = "value")
    ts_train, ts_dev = pre1.train_test
    assert len(ts_test) == int((1 - ratio) * len(ts))
    assert len(ts_train) + len(ts_dev) == int(ratio * len(ts))
    assert len(ts_train) == int(ratio * int(ratio * len(ts)))
    assert len(ts_dev) == int((1 - ratio) * int(ratio * len(ts)))

def test_windows():
    win = 3
    fh = 2
    ts_len = 100
    ts = pd.DataFrame()
    ts["value"] = [i + 1 for i in range(ts_len)]

    pre = Preprocessor(ts, value_col = "value")
    windows, labels = pre.extract_windows(ts["value"].to_numpy(), win, fh)
    assert windows.shape[0] == ts_len - win - fh + 1
    assert windows.shape[0] == labels.shape[0]
    assert windows.shape[1] == win
    assert labels.shape[1] == fh

    assert windows[0][0] == 1
    assert windows[0][1] == 2
    assert windows[0][2] == 3
    assert labels[0][0] == 4
    assert labels[0][1] == 5

    assert windows[-1][0] == 96
    assert windows[-1][1] == 97
    assert windows[-1][2] == 98
    assert labels[-1][0] == 99
    assert labels[-1][1] == 100


def test_window_small_input():
    win = 10
    ts_len = 3
    ts = pd.DataFrame()
    ts["value"] = [i + 1 for i in range(ts_len)]

    pre = Preprocessor(ts, value_col = "value")
    windows, labels = pre.extract_windows(ts["value"].to_numpy(), win)
    print(windows, labels)
    assert windows.shape[0] == 1
    assert windows.shape[0] == labels.shape[0]
    assert windows.shape[1] == win

    assert windows[0][0] == 0
    assert windows[0][-3] == 0
    assert windows[0][-2] == 1
    assert windows[0][-1] == 2
    assert labels[0][0] == 3
