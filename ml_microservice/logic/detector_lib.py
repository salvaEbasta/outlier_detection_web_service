import logging
import os
from typing import Tuple

from ml_microservice import configuration as cfg
from ml_microservice.logic import metadata

class Environment():
    def __init__(self, root_path, 
        assets_dir = cfg.env.assets_dir, 
        temp_dir = cfg.env.temp_dir
    ):
        self.path = root_path
        self.assets_dir = assets_dir
        self.temp_dir = temp_dir
    
    def compose(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        if self.assets_dir not in os.listdir(self.root):
            os.makedirs(self.assets)

    @property
    def root(self):
        return self.path
    
    @property
    def assets(self):
        return os.path.join(self.path, self.assets_dir)
    
    @property
    def temp(self):
        return os.path.join(self.path, self.temp_dir)
    
    def __str__(self):
        return "Env(root:{:s})".format(self.path)

    def __close__(self):
        import shutil
        shutil.rmtree(self.temp)

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
    
    def create_env(self, mID: str) -> Tuple[str, Environment, metadata.Metadata]:
        """
        -> version: str, env: Environment, m: Metadata
        """
        v_num = 0
        if self.has(mID):
            v_num = len(self.versions(mID))
        v = cfg.detectLib.version_format.format(v_num)
        env_ = Environment(self._2storage_path(mID, v))
        env_.compose()
        m = metadata.Metadata().save(env_.root)
        logging.info("New env: {:s}".format(env_))
        return v, env_, m
    
    def retrieve_env(self, mID: str, version: str):
        return Environment(self._2storage_path(mID, version))
    
    def retrieve_metadata(self, mID: str, version: str):
        if not self.has(mID, version):
            return None
        env = Environment(self._2storage_path(mID, version))
        return metadata.load(env.root)
