import os
from importlib import reload
from configparser import ConfigParser

from flask import Flask
from flask import redirect, url_for, abort, Response
from flask import request

from . import service_logic as logic

conf = ConfigParser()
conf.read('config.ini')


def build_app():
    app = Flask(__name__)

    @app.route('/')
    def index():
        return redirect(url_for('models'))

    @app.route('/models', methods=['GET', 'POST'])
    def models():
        try:
            reload(logic)

            if request.method == 'GET':
                return {'available': logic.Toolshed().tools}, 200
            
            elif request.method == 'POST':
                return logic.Toolshed().assemble(request.get_json()), 200
            else:
                abort(405)
        except Exception as e:
            return {'error':type(e), 'description':e.__str__()}, 500

    @app.route('/models/<label>', methods=['GET', 'POST', 'PUT'])
    def model_handling(label):
        try:
            reload(logic)

            model = logic.Toolshed().pickup(dict(name=label))
            
            if request.method == 'GET':
                return model.info, 200
            
            elif request.method == 'POST':
                payload = request.get_json()
                return model.predict(payload['data']), 200
            
            elif request.method == 'PUT':
                payload = request.get_json()
                return model.retrain(), 200
            
            else:
                abort(405)
        except Exception as e:
            print(e)
            return {'error': '{}'.format(type(e)), 'description': str(e)}, 500

    @app.route('/local/datasets')
    def local_sources():
        return {'available': logic.DatasetsLibrary().datasets}

    return app