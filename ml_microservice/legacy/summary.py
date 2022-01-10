from datetime import datetime
import json
import os

from ml_microservice import configuration as cfg

STATUS = dict(active='active', training='under_training')

class Summary():
    def __init__(self, 
                    label = None, 
                    dataset = None, 
                    column = None,
                    status = STATUS["training"], 
                    created_on = datetime.now().isoformat(),
                    train_time = -1,
                ):
        self._status = status
        self._created_on = created_on

        self._train_time = train_time
        self._label = label
        self._dataset = dataset
        self._column = column

    @property
    def training(self):
        return dict(
            total_time = self._train_time,
            label = self._label,
            dataset = self._dataset,
            column = self._column,
        )

    @property
    def values(self):
        return dict(
                status = self._status,
                created_on = self._created_on,
                training = self.training,
            )

    def is_active(self):
        return self._status == STATUS["active"]

    def save(self, ddir):
        f = os.path.join(ddir, cfg.files.detector_summary)
        with open(f, 'w') as f:
            json.dump(self.values, f)

    def load(self, path):
        if not os.path.exists(path):
            self.logger.warning('Can\'t find Summary @{}'.format(path))
            self.__init__()
        else:
            with open(path, 'r') as f:
                tmp = json.load(f)
            self.__init__(
                label = tmp['training']['label'],
                dataset = tmp['training']['dataset'],
                column = tmp['training']['column'],
                train_time = tmp['training']['total_time'],
                status = tmp['status'],
                created_on = tmp['created_on']
            )

    def __repr__(self):
        return "Summary(status: {}, created_on: {}, training: {})".format(
            self._status,
            self._created_on,
            self.training,
        )