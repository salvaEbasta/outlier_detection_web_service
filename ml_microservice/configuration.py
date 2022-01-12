from argparse import Namespace
import os

from ml_microservice.anomaly_detection import configuration as cfg

# Files ------------------------------------------
files = Namespace(
    detector_summary = 'summary.json',
    detector_params = 'params.json',
    detector_history = 'history.csv',
    preprocessing_params = 'preprocessing.json',
)

# Formats -------------------------------------------
formats = Namespace(
    version = "v{:d}",
    identifier_separator = '_',
    joblib_file = "{:s}.joblib",
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
# Metadata ---------------------------------------------
metadata = Namespace(
    default_file = "metadata.json",
)
metadataKs = Namespace(
    status = "status",
    created = "created",
    type = "type",
    ts = "timeserie",
    ts_group = "group",
    ts_dim = "dimension",
    ts_tsID = "tsID",
    train = "training",
    train_trainIDX = "last_train_IDX",
    train_time = "total_time_(s)",
    train_bestConfig = "best_config",
)

# Detector defaults -------------------------------------
detectorDefaults = Namespace(
    max_epochs = 10,
    win_size = 26,
    early_stopping_patience = 3,
    lambda_ = .01,
    k = 3,
)

# DetectorsLib ---------------------------------------------
detectLib = Namespace(
    path = os.path.join("data", "saved"),
    version_format = "v{:d}",
)

env = Namespace(
    assets_dir = "assets",
    temp_dir = "temp"
)

# Trainer defaults ----------------------------------------
detectorTrainer = Namespace(
    path = os.path.join("data", "saved"),
    retrain_patience = 5,
)

# Timeseries ----------------------------------------------
timeseries = Namespace(
    path = os.path.join("data", "timeseries"),
    date_column = cfg.cols["timestamp"],
    value_column = cfg.cols["X"],
    anom_column = cfg.cols["y"],
    pred_prob_column =  cfg.cols["pred_prob"],
    forecast_column = cfg.cols["forecast"],
    nan_str = "null",
)

# Filter --------------------------------------------------
seriesFilter = Namespace(
    min_d_points = 30,
    patience = 10,
)