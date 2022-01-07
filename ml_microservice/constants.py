import os
import configparser
from argparse import Namespace
import shlex
# Files ------------------------------------------
files = Namespace(
    config = 'config.ini',
    detector_summary = 'summary.json',
    detector_params = 'params.json',
    detector_history = 'history.csv',
    preprocessing_params = 'preprocessing.json',
)

# Formats -------------------------------------------
formats = Namespace(
    version = 'v%d',
    identifier_separator = '_',
)

_conf = configparser.ConfigParser()
_conf.read(files.config)

# Logs ------------------------------------------------
log = Namespace(
    path = _conf['Log']['path'],
)
log.controllers = os.path.join(log.path, 'controllers')

# XML ------------------------------------------------
xml = Namespace(
    path = _conf['XML']['path'],
    empty_field_name = _conf['XML']['if_empty_field_name'],
)

splitter = shlex.shlex(_conf['XML']['ids'], posix=True)
splitter.whitespace += ','
splitter.whitespace_split = True
xml.ids = list(splitter)

splitter = shlex.shlex(_conf['XML']['ignore'], posix=True)
splitter.whitespace += ','
splitter.whitespace_split = True
xml.ignore = list(splitter)

# Detector defaults -------------------------------------
detectorDefaults = Namespace(
    max_epochs = int(_conf['AnomalyDetector']['max_epochs']),
    win_size = int(_conf['AnomalyDetector']['window']),
    early_stopping_patience = int(_conf['AnomalyDetector']['patience']),
    lambda_ = float(_conf['AnomalyDetector']['lambda']),
    k = float(_conf['AnomalyDetector']['k']),
)

# Trainer defaults ----------------------------------------
detectorTrainer = Namespace(
    path = _conf['Trainer']['path'],
    retrain_patience = int(_conf['Trainer']['retrain_patience']),
)

# Timeseries ----------------------------------------------
timeseries = Namespace(
    path = _conf['Timeseries']['group_path'],
    date_column = _conf['Timeseries']['date_column'],
)

# Filter --------------------------------------------------
seriesFilter = Namespace(
    min_d_points = int(_conf['SeriesFilter']['min_data_points']),
    patience = int(_conf['SeriesFilter']['serie_patience']),
)

if __name__ == "__main__":
    print(files)
    print(formats)
    print(xml)
    print(detectorDefaults)
    print(detectorTrainer)
    print(seriesFilter)