import os
from configparser import ConfigParser

conf = ConfigParser()
conf.read('config.ini')


def test():
    return {'test': 'test012'}, 200


def models_list():
    return {'models': [file[:-4] for file in os.listdir(conf['Resources']['modelsPath'])]}, 200


def models_train_new(payload: dict):
    return {'error': 'Not implemented'}, 502


def model_info(model:str):
    return {'error': 'Not implemented'}, 502

def model_predict(model: str, payload: dict):
    models = [file[:-4] for file in os.listdir(conf['Resources']['modelsPath'])]
    if model not in models:
        return {'error': 'Model unavailable'}, 404

    # tf.load_model(m)
    return {'request': payload, 'model': model}, 200

def model_retrain(model: str, payload: dict):
    return {'error': 'Not implemented'}, 502
