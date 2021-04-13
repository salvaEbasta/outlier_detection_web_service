import os
from importlib import reload
#import json
#from configparser import ConfigParser

#conf = ConfigParser()
#conf.read('config.ini')

import use_cases

from flask import Flask
from flask import redirect, url_for, abort, Response
from flask import request

app = Flask(__name__)

@app.route('/')
def index():
    return redirect(url_for('models'))

@app.route('/models', methods=['GET', 'POST'])
def models():
    try:
        reload(use_cases)
        if request.method == 'GET':
            return use_cases.models_list()
        elif request.method == 'POST':
            return use_cases.models_train_new(request.get_json())
        else:
            abort(405)
    except Exception as e:
        return {'error':type(e), 'description':e.__str__()}, 500

@app.route('/models/<model>', methods=['GET', 'POST', 'PUT'])
def model_handling(model):
    try:
        reload(use_cases)
        if request.method == 'GET':
            return use_cases.model_info(model)
        elif request.method == 'POST':
            return use_cases.model_predict(model, request.get_json())
        elif request.method == 'PUT':
            return use_cases.model_retrain(model, request.get_json())
        else:
            abort(405)
    except Exception as e:
        return {'error':type(e), 'description':e.__str__()}, 500

@app.route('/test')
def route_test():
    reload(use_cases)
    return use_cases.test()

if __name__ == '__main__':
    app.run()