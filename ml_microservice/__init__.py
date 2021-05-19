import os
from importlib import reload
from configparser import ConfigParser

from flask import Flask
from flask import redirect, url_for, Response
from flask import request

from . import service_logic as logic
from .anomaly_detection import model_factory
from . import strings

conf = ConfigParser()
conf.read(strings.config_file)


def build_app():
    app = Flask(__name__)

    @app.route('/')
    def index():
        return redirect(url_for('series'))

    @app.route('/api/series', methods=['GET', 'POST'])
    def series():
        reload(logic)
        try:
            if request.method == 'GET':
                return {'available': logic.DetectorsLibrary().list}, 200

            elif request.method == 'POST':
                print("Debug: in controller")
                print(f"R: {request.get_json()}")
                return logic.DetectorsLibrary().assemble(request.get_json()), 201

            else:
                return {'error': 'MethodNotAllowed'}, 405
        except Exception as e:
            print(e)
            return {'error':type(e), 'description':e.__str__()}, 500

    @app.route('/api/series/<label>/<version>', methods=['GET', 'POST', 'PUT'])
    def anomaly_detection(label, version):
        reload(logic)
        try:
            anomaly_controller = logic.AnomalyDetection(label=label, version=version)
            if request.method == 'GET':
                return anomaly_controller.info, 200

            elif request.method == 'POST':
                print("Debug: in controller")
                payload = request.get_json()
                if 'data' not in payload:
                    raise ValueError('The payload is missing field \'data\'')
                elif not anomaly_controller.detector_ready:
                    return {'error':'NotAcceptable', 'description':'The model selected is currently under training'}, 406
                else:
                    if 'epochs' in payload and payload['epochs'] > 0:
                        return anomaly_controller.predict(payload['data'], epochs=payload['epochs']), 200
                    else:
                        return anomaly_controller.predict(payload['data']), 200
            
            elif request.method == 'PUT':
                print("Debug: in controller")
                payload = request.get_json()
                if 'data' not in payload:
                    raise ValueError('The payload is missing field \'data\'')
                elif not anomaly_controller.detector_ready:
                    return {'error':'NotAcceptable', 'description':'The model selected is currently under training'}, 406
                else:
                    if 'epochs' in payload and payload['epochs'] > 0:
                        return anomaly_controller.update(payload['data'], epochs=payload['epochs']), 200
                    else:
                        return anomaly_controller.update(payload['data']), 200

            else:
                return {'error': 'MethodNotAllowed'}, 405
        except ValueError as e:
            print(e)
            return {'error':'BadRequest', 'description': str(e)}, 400
        except Exception as e:
            print(e)
            return {'error': '{}'.format(type(e)), 'description': str(e)}, 500

    @app.route('/api/datasets/local')
    def local_datasets():
        reload(logic)
        return {'available': logic.DatasetsLibrary().datasets}

    @app.route('/api/regressors')
    def anomaly_detector_regressors():
        reload(model_factory)
        return {'available': model_factory.RegressorFactory().available()}, 200
    
    return app