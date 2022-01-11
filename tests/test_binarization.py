from pickle import dump
import joblib
import os

import numpy as np

from tests import TEST_DIR
from ml_microservice import configuration as old_cfg
from ml_microservice.anomaly_detection.binarization import empirical_rule as emp_r
from ml_microservice.anomaly_detection.loaders import EmpRuleLoader

def test_emp_rule_save_n_load():
    file = old_cfg.empRule.file_ext.format("er")
    path = os.path.join(TEST_DIR, file)
    if os.path.exists(path):
        os.remove(path)
    shape0 = 10
    X = np.random.normal(size=[shape0,])
    y = np.random.normal(size=[shape0,])
    
    er = emp_r.EmpiricalRule()
    er.fit(X, y)
    er.save(TEST_DIR)

    tmp = EmpRuleLoader().load(TEST_DIR)
    assert tmp._estimator_type == er._estimator_type
    assert tmp.mean_ == er.mean_
    assert tmp.var_ == er.var_
    os.remove(path)

def test_emp_rule():
    shape0 = 20
    X = np.random.normal(size=[shape0,])
    y = np.ones(shape=[shape0,])
    y = 4*y
    er = emp_r.EmpiricalRule()
    er.fit(X)
    assert er.mean_ < 0.1
    assert 0.5 < er.var_ < 1.1
    y_hat = er.classify(y)
    assert y_hat.shape[0] == shape0
    assert all(y_hat > 0)
    