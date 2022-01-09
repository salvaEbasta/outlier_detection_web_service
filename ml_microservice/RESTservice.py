import os
from importlib import reload
import logging

from flask import Flask
from flask import redirect, url_for
from flask import request
from flask import json
from werkzeug.exceptions import HTTPException

from . import controllers
from . import configuration as cfg


def build_app():
    log_file = os.path.join(cfg.log.path, 'microservice.log')
    if not os.path.exists(log_file):
        os.makedirs(os.path.split(log_file)[0])
    
    logging.basicConfig(filename = log_file)
    
    app = Flask(__name__)

    @app.route('/')
    def index():
        return redirect(url_for('anomaly_detectors'))

    # AnomalyDetectors
    @app.route('/api/anomaly_detectors', methods=['GET', 'POST'])
    def anomaly_detectors():
        reload(controllers)
        logging.info('In Anomaly Detector')
        if request.method == 'GET':
            return controllers.ListDetectors().handle()
        elif request.method == 'POST':
            logging.info(f"Anomaly detectors: {request}")
            return controllers.NewDetector(request=request).handle()

    @app.route('/api/anomaly_detectors/<mID>/<version>', methods=['GET', 'POST'])
    def detector(mID, version):
        logging.info('detector: {:s}.{:s}[{:s}]'.format(mID, version, request.method))
        reload(controllers)
        if request.method == 'GET':
            return controllers.ShowDetector(mID, version).handle()
        elif request.method == 'POST':
            return controllers.Detect(
                identifier = mID, 
                version = version, 
                request = request
            ).handle()
    
    @app.route('/api/anomaly_detectors/<mID>/<version>/history')
    def detectors_history(mID, version):
        logging.info('detector history: {:s}.{:s}'.format(mID, version))
        reload(controllers)
        return controllers.ShowDetectorHistory(mID, version).handle()
    
    @app.route('/api/anomaly_detectors/<mID>/<version>/parameters')
    def detectors_params(mID, version):
        logging.info('detector params: {:s}.{:s}'.format(mID, version))
        reload(controllers)
        return controllers.ShowDetectorParameters(mID, version).handle()

    # Xml
    @app.route('/api/conversion/xml', methods=['POST'])
    def dump_xml():
        logging.info('Convert xml')
        reload(controllers)
        return controllers.ConvertXML(request=request).handle()

    # TimeSeriesForecasting
    @app.route('/api/time_series_forecasting/models')
    def forecasting_models():
        logging.info('tsf/models')
        reload(controllers)
        return controllers.ListForecasters().handle()

    # Datasets
    @app.route('/api/timeseries')
    def local_datasets():
        logging.info('timeseries: ')
        reload(controllers)
        return controllers.ListDatasets().handle()

    @app.route('/api/timeseries/<group>/<dimension>')
    def dataset_exploration(group, dimension):
        logging.info('explore dimension: {:s}.{:s}'.format(group, dimension))
        reload(controllers)
        return controllers.ExploreDataset(
            group = group, 
            dimension = dimension
        ).handle()

    @app.route('/api/timeseries/<group>/<dimension>/<tsID>')
    def column_exploration(group, dimension, tsID):
        logging.info('explore timeseries: {:s}.{:s}.{:s}'.format(group, dimension, tsID))
        reload(controllers)
        return controllers.ExploreColumn(
            group = group, 
            dimension = dimension, 
            tsID = tsID
        ).handle()

    # Generic HTTP error handling
    @app.errorhandler(HTTPException)
    def error_handler(e):
        response = e.get_response()
        response.data = json.dumps({
            "code": e.code,
            "name": e.name,
            "description": e.description,
        })
        response.content_type = "application/json"
        return response

    return app