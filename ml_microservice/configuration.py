from argparse import Namespace
import os

# Files ------------------------------------------
files = Namespace(
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

# Logs ------------------------------------------------
log = Namespace(
    path = os.path.join("data", "log"),
)
log.controllers = os.path.join(log.path, 'controllers')

# XML ------------------------------------------------
xml = Namespace(
    path = os.path.join("data", "xml"),
    empty_field_name = "not_specified",
    ids = ["giorno_settimana", ".*_cd", ".*_CD", "Abc_articolo"],
    ignore = [".*_ds", ".*_DS"],
)

# Detector defaults -------------------------------------
detectorDefaults = Namespace(
    max_epochs = 10,
    win_size = 26,
    early_stopping_patience = 3,
    lambda_ = .01,
    k = 3,
)

# Trainer defaults ----------------------------------------
detectorTrainer = Namespace(
    path = os.path.join("data", "saved"),
    retrain_patience = 5,
)

# Timeseries ----------------------------------------------
timeseries = Namespace(
    path = os.path.join("data", "timeseries"),
    date_column = "Date",
    value_column = "Value",
)

# Filter --------------------------------------------------
seriesFilter = Namespace(
    min_d_points = 30,
    patience = 10,
)

if __name__ == "__main__":
    print(files)
    print(formats)
    print(xml)
    print(detectorDefaults)
    print(detectorTrainer)
    print(seriesFilter)