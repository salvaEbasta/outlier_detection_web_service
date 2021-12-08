import os
from importlib import reload
import logging

from flask import Flask
from flask import redirect, url_for, Response
from flask import request
from flask import json
from werkzeug.exceptions import HTTPException

from . import controllers
from . import service_logic as logic
from .anomaly_detection import model_factory
from . import constants


def build_app():
    logging.basicConfig(filename=os.path.join(constants.log.path, 'microservice.log'))
    
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

    @app.route('/api/anomaly_detectors/<label>/<version>', methods=['GET', 'POST'])
    def detector(label, version):
        logging.info('detector: {:s}.{:s}[{:s}]'.format(label, version, request.method))
        reload(controllers)
        if request.method == 'GET':
            return controllers.ShowDetector(label, version).handle()
        elif request.method == 'POST':
            return controllers.Detect(
                identifier = label, 
                version = version, 
                request = request
            ).handle()
    
    @app.route('/api/anomaly_detectors/<label>/<version>/history')
    def detectors_history(label, version):
        logging.info('detector history: {:s}.{:s}'.format(label, version))
        reload(controllers)
        return controllers.ShowDetectorHistory(label, version).handle()
    
    @app.route('/api/anomaly_detectors/<label>/<version>/parameters')
    def detectors_params(label, version):
        logging.info('detector params: {:s}.{:s}'.format(label, version))
        reload(controllers)
        return controllers.ShowDetectorParameters(label, version).handle()

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
    @app.route('/api/datasets/local')
    def local_datasets():
        logging.info('datasets local')
        reload(controllers)
        return controllers.ListDatasets().handle()

    @app.route('/api/datasets/local/<label>/<dataset>')
    def dataset_exploration(label, dataset):
        logging.info('explore dataset: {:s}.{:s}'.format(label, dataset))
        reload(controllers)
        return controllers.ExploreDataset(label=label, dataset=dataset).handle()

    @app.route('/api/datasets/local/<label>/<dataset>/<column>')
    def column_exploration(label, dataset, column):
        logging.info('explore column: {:s}.{:s}.{:s}'.format(label, dataset, column))
        reload(controllers)
        return controllers.ExploreColumn(label=label, dataset=dataset, column=column).handle()

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