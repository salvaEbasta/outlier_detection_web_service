import os

import numpy as np
import pandas as pd

from tests import TEST_DIR
from ml_microservice.anomaly_detection import configuration as cfg
from ml_microservice.anomaly_detection.residual_analysis import empirical_rule as emp_r
from ml_microservice.anomaly_detection.loaders import EmpRuleLoader

def test_emp_rule_save_n_load():
    file = cfg.empRule["file_ext"].format("er")
    file_path = os.path.join(TEST_DIR, file)
    if os.path.exists(file_path):
        os.remove(file_path)
    shape0 = 10
    X = pd.DataFrame()
    X[cfg.cols["X"]] = np.random.normal(size=[shape0,])
    
    er = emp_r.EmpiricalRule()
    er.fit(X)
    er.save(TEST_DIR)

    tmp = EmpRuleLoader().load(TEST_DIR)
    assert tmp.mean_ == er.mean_
    assert tmp.var_ == er.var_
    os.remove(file_path)

def test_emp_rule():
    shape0 = 20
    ts = pd.DataFrame()
    ts[cfg.cols["X"]] = np.random.normal(size=[shape0,])
    er = emp_r.EmpiricalRule()
    er.fit(ts)
    assert er.mean_ < 0.1
    print(er.var_)
    assert 0.5 < er.var_ < 1.1
    
    ts[cfg.cols["X"]] = 4 * np.ones(shape=[shape0,])
    y_hat = er.predict(ts)
    assert len(y_hat) == shape0
    assert all([v > 0 for v in y_hat[cfg.cols["X"]]])
    