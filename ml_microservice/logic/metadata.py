import datetime
import json
import os
from typing import Dict

from ml_microservice import configuration as cfg

class Status:
    training = {
        "name": "under_training",
        "code": -1
    }
    trained = {
        "name": "trained",
        "code": 0
    }

def is_valid(content):
    return cfg.metadataKs.status in content and \
            cfg.metadataKs.created in content and \
            cfg.metadataKs.type in content and \
            cfg.metadataKs.ts in content and \
            cfg.metadataKs.train in content

def load(dir_path):
    if not os.path.exists(dir_path):
        return None
    if cfg.metadata.default_file not in os.listdir(dir_path):
        return None
    m_path = os.path.join(dir_path, cfg.metadata.default_file)
    with open(m_path, "r") as f:
        content = json.load(f)
    if not is_valid(content):
        return None
    meta = Metadata(metadata = content)
    return meta

class Metadata():
    def __init__(self, metadata = None):
        if metadata is not None:
            self._content = metadata
        else:
            self._content = {}
            self._content[cfg.metadataKs.status] = Status.training
            self._content[cfg.metadataKs.created] = datetime.now().isoformat()
            self._content[cfg.metadataKs.type] = ""
            self._content[cfg.metadataKs.ts] = {}
            self._content[cfg.metadataKs.train] = {}
    
    def to_dict(self):
        return self._content

    def set_type(self, type: str):
        self._content[cfg.metadataKs.type] = type
    
    @property
    def model_type(self):
        return self._content[cfg.metadataKs.type]
    
    def set_training_info(self, last_train_IDX: int, total_time: float, 
                            best_config: Dict, last_dev_IDX: int = -1, ):
        self._content[cfg.metadataKs.status] = Status.trained
        tmp = {}
        tmp[cfg.metadataKs.train_trainIDX] = last_train_IDX
        tmp[cfg.metadataKs.train_time] = total_time
        tmp[cfg.metadataKs.train_bestConfig] = best_config
        if 0 <= last_dev_IDX:
            tmp[cfg.metadataKs.train_devIDX] = last_dev_IDX
    
    @property
    def last_dev_IDX(self):
        devIDX_key = cfg.metadataKs.train_devIDX
        if devIDX_key not in self._content[cfg.metadataKs.train]:
            return -1
        return self._content[cfg.metadataKs.train][devIDX_key]

    
    def get_ts(self):
        if not len(self._content[cfg.metadataKs.ts]):
            return None
        return self._content[cfg.metadataKs.ts][cfg.metadataKs.ts_group], \
            self._content[cfg.metadataKs.ts][cfg.metadataKs.ts_dim], \
            self._content[cfg.metadataKs.ts][cfg.metadataKs.ts_tsID], \

    def set_ts(self, group: str, dimension: str, tsID: str):
        tmp = {}
        tmp[cfg.metadataKs.ts_group] = group
        tmp[cfg.metadataKs.ts_dim] = dimension
        tmp[cfg.metadataKs.ts_tsID] = tsID
        self._content[cfg.metadataKs.ts] = tmp

    def is_training_done(self):
        return self._content[cfg.metadataKs.type]["code"] == Status.trained["code"]

    def save(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file = os.path.join(dir_path, cfg.metadata.file)
        with open(file, "w") as f:
            json.dump(self._content, f, indent = 4)

    def __str__(self) -> str:
            return json.dumps(self._content)