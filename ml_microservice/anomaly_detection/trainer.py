import os
import time
from kerastuner.engine.tuner import Tuner

import pandas as pd

from ml_microservice import configuration as cfg
from ml_microservice.anomaly_detection.factory import Factory
from ml_microservice.anomaly_detection.loaders import Loader
from ml_microservice.logic.metadata import Metadata

class Trainer():
    def __init__(self, env: str):
        self.env = env
    
    @property
    def assets_location(self):
        return os.path.join(self.env, cfg.trainer.assets_dir)

    def train(self, tuner: Tuner, ts: pd.DataFrame, metadata: Metadata):
        t0 = time.time()



        metadata = Metadata()
        # preprocessing
        # tuning
        deltaT = time.time() - t0
        metadata.set_training_info(
            total_time = deltaT,
        )

        metadata.save(self.env)
        pass

    def predict(self, loader: Loader, ts: pd.DataFrame):
        # load
        model = loader.load(self.env)
        model
        # predict
        # evaluate
        # retrain
        pass
