import logging
import os

from ml_microservice import configuration as cfg
from ml_microservice.logic import metadata

class AnomalyDetectorsLibrary():
    def __init__(self, storage = cfg.detectLib.path):
        self.logger = logging.getLogger('detectLib')
        self.logger.setLevel(logging.INFO)
        self.storage = storage
    
    def _2storage_path(self, mID: str, version: str = None):
        label_path = os.path.join(self.storage, mID)
        if version is None:
            return label_path
        version_path = os.path.join(label_path, version)
        return version_path
    
    def modelIDS(self):
        return [mID for mID in os.listdir(self.storage)
                    if os.path.isdir(self._2storage_path(mID))]

    def versions(self, mID: str):
        if not os.path.exists(self._2storage_path(mID)):
            return []
        return [v for v in os.listdir(self._2storage_path(mID))
                    if os.path.isdir(self._2storage_path(mID, v))]

    def list(self):
        return [{"mID": mID, "versions": self.versions(mID)} 
                    for mID in self.modelIDS()]
    
    def has(self, mID: str, version: str = None):
        has_mID = mID in os.listdir(self.storage) and \
            os.path.isdir(self._2storage_path(mID))
        if version is None:
            return has_mID
        return has_mID and \
            version in os.listdir(self._2storage_path(mID)) and \
            os.path.isdir(self._2storage_path(mID, version))
    
    def create_env(self, mID: str):
        v_num = 0
        if self.has(mID):
            v_num = len(self.versions(mID))
        v = cfg.detectLib.version_format.format(v_num)
        env_ = self._2storage_path(mID, v)
        os.makedirs(env_)
        logging.info("New env: {:s}".format(env_))
        return v, env_
    
    def retrieve_env(self, mID: str, version: str):
        return self._2storage_path(mID, version)
    
    def retrieve_metadata(self, mID: str, version: str):
        if not self.has(mID, version):
            return None
        env = self._2storage_path(mID, version)
        return metadata.load(env)
