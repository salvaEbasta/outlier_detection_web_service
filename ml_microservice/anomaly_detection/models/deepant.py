from ..detector import AnomalyDetector
from .windowed_gaussian import WindowedGaussian

class DeepAnT(AnomalyDetector):
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self, ts):
        predictor = None
        WindowedGaussian()
        return self
    
    def predict(self, ts):
        return None