import os
#from importlib import reload
import logging

from flask import Flask
from flask import redirect, url_for
from flask import request
from flask import json
from werkzeug.exceptions import HTTPException

from . import controllers
from . import configuration as old_cfg


def build_app():
    log_file = os.path.join(old_cfg.log.path, 'microservice.log')
    if not os.path.exists(log_file):
        os.makedirs(os.path.split(log_file)[0])
    
    logging.basicConfig(filename = log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    app = Flask(__name__)

    @app.route('/')
    def index():
        return redirect(url_for('anomaly_detection'))

    # AnomalyDetection
    @app.route('/api/anomaly_detection', methods=['GET', 'POST'])
    def anomaly_detection():
        #reload(controllers)
        logging.info('In Anomaly Detector')
        if request.method == 'GET':
            return controllers.ListSavedDetectors().handle()
        elif request.method == 'POST':
            logging.info(f"Anomaly detectors: {request}")
            return controllers.DetectorTrain(request=request).handle()
    
    @app.route('/api/anomaly_detection/methods')
    def list_models():
        logging.info('anom_detect/methods')
        #reload(controllers)
        return controllers.ListMethods().handle()
    
    @app.route('/api/anomaly_detection/<mID>/<version>', methods=['GET', 'POST'])
    def detector(mID, version):
        logging.info('detector: {:s}.{:s}[{:s}]'.format(mID, version, request.method))
        #reload(controllers)
        if request.method == 'GET':
            return controllers.DetectorMetadata(mID, version).handle()
        elif request.method == 'POST':
            return controllers.DetectPredict(
                identifier = mID, 
                version = version, 
                request = request
            ).handle()
    
    @app.route('/api/anomaly_detection/<mID>/<version>/evaluate', methods=['GET', 'POST'])
    def detector_evaluation(mID, version):
        logging.info('detector: {:s}.{:s}[{:s}]'.format(mID, version, request.method))
        #reload(controllers)
        if request.method == 'GET':
            return controllers.DetectorEvaluate(mID, version).handle()
        elif request.method == 'POST':
            c = controllers.DetectorEvaluate(mID, version)
            c.set_request(request)
            return c.handle()

    @app.route('/api/anomaly_detection/<mID>/<version>/history')
    def detectors_history(mID, version):
        logging.info('detector history: {:s}.{:s}'.format(mID, version))
        #reload(controllers)
        return controllers.ShowDetectorHistory(mID, version).handle()
    
    @app.route('/api/anomaly_detection/<mID>/<version>/parameters')
    def detectors_params(mID, version):
        logging.info('detector params: {:s}.{:s}'.format(mID, version))
        #reload(controllers)
        return controllers.DetectorParameters(mID, version).handle()

    # Xml
    @app.route('/api/convert/xml', methods=['POST'])
    def dump_xml():
        logging.info('Convert xml')
        #reload(controllers)
        return controllers.ConvertXML(request=request).handle()

    # Datasets
    @app.route('/api/timeseries')
    def list_timeseries():
        logging.info('timeseries: ')
        #reload(controllers)
        return controllers.ListTimeseries().handle()

    @app.route('/api/timeseries/<groupID>/<dimID>')
    def explore_dimension(groupID, dimID):
        logging.info('explore dimID: {:s}.{:s}'.format(groupID, dimID))
        #reload(controllers)
        return controllers.ExploreTSDim(
            groupID = groupID, 
            dimID = dimID
        ).handle()

    @app.route('/api/timeseries/<groupID>/<dimID>/<tsID>')
    def explore_timeseries(groupID, dimID, tsID):
        logging.info('explore timeseries: {:s}.{:s}.{:s}'.format(groupID, dimID, tsID))
        #reload(controllers)
        return controllers.ExploreTS(
            groupID = groupID, 
            dimID = dimID, 
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