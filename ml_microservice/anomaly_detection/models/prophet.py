import inspect
import json
import os

import numpy as np
import pandas as pd

from .. import configuration as cfg
from ..transformers import Preprocessor
from ..detector import AnomalyDetector, Forecaster
from .windowed_gaussian import WindowedGaussian
from ..metrics import naive_prediction

class Prophet(AnomalyDetector, Forecaster):
    def __init__(self, gauss_win = 32, gauss_step = 16, win = 32, 
                    size1 = 128, dropout1 = .45, rec_dropout1 = .45,
                    size2 = 128, dropout2 = .45, rec_dropout2 = .45
                    ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val) 

        self.preload = None
        self.forecaster = None
        self.classifier = None