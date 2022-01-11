import datetime
import json
import os
from typing import Dict

from ml_microservice import configuration as old_cfg

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
    return old_cfg.metadataKs.status in content and \
            old_cfg.metadataKs.created in content and \
            old_cfg.metadataKs.type in content and \
            old_cfg.metadataKs.ts in content and \
            old_cfg.metadataKs.train in content

def load(dir_path):
    if not os.path.exists(dir_path):
        return None
    if old_cfg.metadata.default_file not in os.listdir(dir_path):
        return None
    m_path = os.path.join(dir_path, old_cfg.metadata.default_file)
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
            self._content[old_cfg.metadataKs.status] = Status.training
            self._content[old_cfg.metadataKs.created] = datetime.now().isoformat()
            self._content[old_cfg.metadataKs.type] = ""
            self._content[old_cfg.metadataKs.ts] = {}
            self._content[old_cfg.metadataKs.train] = {}
    
    def to_dict(self):
        return self._content

    def set_type(self, type: str):
        self._content[old_cfg.metadataKs.type] = type
    
    @property
    def model_type(self):
        return self._content[old_cfg.metadataKs.type]
    
    def set_training_info(self, last_train_IDX: int, total_time: float, 
                            best_config: Dict, ):
        self._content[old_cfg.metadataKs.status] = Status.trained
        tmp = {}
        tmp[old_cfg.metadataKs.train_trainIDX] = last_train_IDX
        tmp[old_cfg.metadataKs.train_time] = total_time
        tmp[old_cfg.metadataKs.train_bestConfig] = best_config
    
    @property
    def last_train_IDX(self):
        return self._content[old_cfg.metadataKs.train][old_cfg.metadataKs.train_trainIDX]

    
    def get_ts(self):
        if not len(self._content[old_cfg.metadataKs.ts]):
            return None
        return self._content[old_cfg.metadataKs.ts][old_cfg.metadataKs.ts_group], \
            self._content[old_cfg.metadataKs.ts][old_cfg.metadataKs.ts_dim], \
            self._content[old_cfg.metadataKs.ts][old_cfg.metadataKs.ts_tsID], \

    def set_ts(self, group: str, dimension: str, tsID: str):
        tmp = {}
        tmp[old_cfg.metadataKs.ts_group] = group
        tmp[old_cfg.metadataKs.ts_dim] = dimension
        tmp[old_cfg.metadataKs.ts_tsID] = tsID
        self._content[old_cfg.metadataKs.ts] = tmp

    def is_training_done(self):
        return self._content[old_cfg.metadataKs.type]["code"] == Status.trained["code"]

    def save(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file = os.path.join(dir_path, old_cfg.metadata.file)
        with open(file, "w") as f:
            json.dump(self._content, f, indent = 4)

    def __str__(self) -> str:
            return json.dumps(self._content)