
cols = {
    "timestamp": "timestamp",
    "X": "value",
    "y": "outlier",
}

# Detector ------------------------------------------
anomDetect = {
    "file_name": "model",
    "file_ext": "{:s}.pkl",
}

windGauss = {
    "file_ext": anomDetect["file_ext"].format("{:s}.wingauss"),
}
windGauss["default_file"] = windGauss["file_ext"].format("wg")

empRule = {
    "file_ext": anomDetect["file_ext"].format("{:s}.emprule"),
}
empRule["default_file"] = empRule["file_ext"].format("er")

# Factory -------------------------------------------------
factory = {
    "tuner_k": "tuner",
    "loader_k": "loader",
}

# Tuner --------------------------------------------------
tuner = {
    "results_file": "explored_config.json"
}

# Evaluator ----------------------------------------------
evaluator = {
    "history_file": "history.csv",
}

# Transformer --------------------------------------------
preprocessor = {
    "good_nan_ratio": 0.3
}