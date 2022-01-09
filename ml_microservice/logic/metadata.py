import datetime
import json
import os

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

def load(metadata_path):
    if os.path.split(metadata_path)[-1] != cfg.metadata.file:
        return None
    with open(metadata_path, "r") as f:
        content = json.load(f)
    if not is_valid(content):
        return None
    meta = Metadata(metadata = content)
    return meta

class Metadata():
    def __init__(self, metadata = None):
        if metadata is not None:
            self.content = metadata
        else:
            self.content = {}
            self.content[cfg.metadataKs.status] = Status.training
            self.content[cfg.metadataKs.created] = datetime.now().isoformat()
            self.content[cfg.metadataKs.type] = ""
            self.content[cfg.metadataKs.ts] = {}
            self.content[cfg.metadataKs.train] = {}
    
    def set_type(self, type: str):
        self.content[cfg.metadataKs.type] = type
    
    def set_training_info(self, last_train_IDX: int, total_time: float,
                            last_dev_IDX: int = -1,):
        self.content[cfg.metadataKs.status] = Status.trained
        tmp = {}
        tmp[cfg.metadataKs.train_trainIDX] = last_train_IDX
        tmp[cfg.metadataKs.train_time] = total_time
        if 0 <= last_dev_IDX:
            tmp[cfg.metadataKs.train_devIDX] = last_dev_IDX
    
    def get_ts(self):
        if not len(self.content[cfg.metadataKs.ts]):
            return None
        return self.content[cfg.metadataKs.ts][cfg.metadataKs.ts_group], \
            self.content[cfg.metadataKs.ts][cfg.metadataKs.ts_dim], \
            self.content[cfg.metadataKs.ts][cfg.metadataKs.ts_tsID], \

    def set_ts(self, group: str, dimension: str, tsID: str):
        tmp = {}
        tmp[cfg.metadataKs.ts_group] = group
        tmp[cfg.metadataKs.ts_dim] = dimension
        tmp[cfg.metadataKs.ts_tsID] = tsID
        self.content[cfg.metadataKs.ts] = tmp

    def is_training_done(self):
        return self.content[cfg.metadataKs.type]["code"] == Status.trained["code"]

    def __str__(self) -> str:
        return json.dumps(self.content)

    def save(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file = os.path.join(dir_path, cfg.metadata.file)
        with open(file, "w") as f:
            json.dump(self.content, f, indent = 4)
