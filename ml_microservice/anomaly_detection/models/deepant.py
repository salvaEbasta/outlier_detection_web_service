from ..detector import AnomalyDetector, Forecaster
from .windowed_gaussian import WindowedGaussian

class DeepAnT(AnomalyDetector, Forecaster):
    def __init__(self):
        pass
    
    def forecast(self, ts):
        return None

    def fit(self, ts):
        predictor = None
        WindowedGaussian()
        return self
    
    def predict(self, ts):
        return None