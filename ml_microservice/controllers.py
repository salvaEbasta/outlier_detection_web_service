from importlib import reload
import os
from typing import List
import json
import logging

import numpy as np

from ml_microservice import constants
from ml_microservice import service_logic as logic
from ml_microservice.anomaly_detection import model_factory, detector
from ml_microservice.conversion import Xml2Csv

class Controller():
    def __init__(self, lvl = logging.DEBUG):
        reload(logic)
        reload(constants)
        reload(detector)
        log_id = str(self.__class__.__name__).lower()
        self.logger = logging.getLogger(log_id)
        self.logger.setLevel(lvl)

    def handle(self, *args, **kwargs):
        try:
            return self._handle(*args, **kwargs)
        except Exception as e:
            return {
                    'code': 500,
                    'name': type(e),
                    'description': str(e),
            }, 500

    def _handle(self, *args, **kwargs):
        raise NotImplementedError()

class ConvertXML(Controller):
    def __init__(self, request):
        super().__init__()
        self.request = request
        self.logger.debug(f"{request.get_json()}")
        self.payload = request.get_json()

    def _unpack(self, payload):
        self.logger.info("Unpack: {}".format(payload))
        self.xml_str, self.label = '', ''

        # Mandatory
        if 'xml' not in payload:
            raise ValueError('The field xml must be specified')
        elif 'label' not in payload:
            raise ValueError('The field label must be specified')
        self.xml_str = payload['xml']
        self.label = payload['label']

        # Optionals
        if 'field_name_id' in payload and type(payload['field_name_id']) is not str:
            raise ValueError('The field field_name_id must be a string')
        self.field_id = payload.get('field_name_id', None)
        
        if 'field_name_ignore' in payload and type(payload['field_name_ignore']) is not List[str]:
            raise ValueError('The field field_name_ignore must be specified a string')
        self.field_ignore = payload.get('field_name_ignore', None)

        if 'store' in payload and type(payload['store']) is not bool:
            raise ValueError('The field store must be a boolean')
        self.store = payload.get('store', False)

    def _handle(self, *args, **kwargs):
        try:
            self._unpack(self.payload)
            dl = logic.DatasetsLibrary()
            parsings = dl.convert(
                label = self.label,
                xml = self.xml_str,
                field_id = self.field_id,
                field_ignore = self.field_ignore,
                store = self.store,
            )
            self.logger.debug("Parsings: {}".format(parsings))
            return {
                'code': 200,
                'stored': self.store,
                'extracted': [{'dimension': name, 'data': data} for name, data in parsings],
            }, 200
        except ValueError as e:
            return {
                'code': 400,
                'name': 'BadeRequest',
                'description': str(e),
            }, 400

class ListDatasets(Controller):
    def __init__(self):
        super().__init__()

    def _handle(self, *args, **kwargs):
        return {
            'code': 200,
            'available': logic.DatasetsLibrary().datasets,
        }, 200

class ExploreDataset(Controller):
    def __init__(self, label, dataset):
        super().__init__()
        self.label = label
        self.dataset = dataset

    def _handle(self, *args, **kwargs):
        dset_lib = logic.DatasetsLibrary()
        if not dset_lib.has_label(self.label):
            return {
                'code': 404,
                'error': 'NotFound',
                'description': 'Can\'t find label \'{:s}\''.format(self.label),
            }, 404
        elif not dset_lib.has(self.label, self.dataset):
            return {
                'code': 404,
                'error': 'NotFound',
                'description': 'Can\'t find the dataset \'{:s}\' under \'{:s}\''.format(self.dataset, self.label),
            }, 404
        else:
            dframe = dset_lib.fetch(self.label, self.dataset)
            return {
                'code': 200,
                'columns': list(dframe.columns),
                'shape': list(dframe.values.shape),
            }, 200

class ExploreColumn(Controller):
    def __init__(self, label, dataset, column):
        super().__init__()
        self.label = label
        self.dataset = dataset
        self.column = column

    def _handle(self, *args, **kwargs):
        dset_lib = logic.DatasetsLibrary()
        try:
            if not dset_lib.has_label(self.label):
                raise ValueError('Can\'t find label \'{:s}\''.format(self.label))
            elif not dset_lib.has(self.label, self.dataset):
                raise ValueError('Can\'t find the dataset \'{:s}\' under \'{:s}\''.format(self.dataset, self.label))
            else:
                values, col_name = dset_lib.fetch_column(self.label, self.dataset, self.column)
                return {
                    'code': 200,
                    'column_name': col_name,
                    'values': np.array(values).tolist(),
                }, 200
        except ValueError as e:
            return {
                'code': 404,
                'name': 'NotFound',
                'description': str(e),
            }, 404

class ListForecasters(Controller):
    def __init__(self):
        super().__init__()

    def _handle(self, *args, **kwargs):
        return {
            'code': 200,
            'available': model_factory.ForecasterFactory().available(),
        }, 200

class ListDetectors(Controller):
    def __init__(self):
        super().__init__()

    def _handle(self, *args, **kwargs):
        tmp = logic.DetectorTrainer().detectors_list
        return {
            'code': 200,
            'detectors': tmp,
        }, 200

class NewDetector(Controller):
    def __init__(self, request):
        super().__init__()
        self.logger.info(f"Request: {request.get_json()}")
        self.request = request
        self.payload = request.get_json()
        self.logger.debug("End __init__")

    def _unpack(self, payload):
        self.identifier = None
        self.training = None
        self.forecasting = None
        
        self.logger.debug('Payload: check \'identifier\'')
        if 'identifier' not in payload:
            raise ValueError('The field \'identifier\' must be specified')
        self.identifier = payload['identifier']
        if type(self.identifier) is not str:
            raise ValueError('The identifier must be a string')
        
        self.logger.debug('Payload: check \'training\'')
        if 'training' not in payload:
            raise ValueError('A field \'training\' must be specified. Must contain a label, a dataset and a column')
        else:
            self.training = dict()

            if 'label' not in payload['training']:
                raise ValueError('A field \'label\' in \'training\' must be specified')
            else:
                self.training['label'] = payload['training']['label']
                if type(self.training['label']) is not str:
                    raise ValueError('The training label must be a string')

            if 'dataset' not in payload['training']:
                raise ValueError('A field \'dataset\' in \'training\' must be specified')
            else:
                self.training['dataset'] = payload['training']['dataset']
                if type(self.training['dataset']) is not str:
                    raise ValueError('The training dataset must be a string')
        
            if 'column' in payload:
                self.training['column'] = payload['training']['column']
                if type(self.training['column']) is not str:
                    raise ValueError('The training column must be a string')
            else:
                self.training['column'] = None

        self.logger.debug('Payload: check \'forecasting_model\'')
        if 'forecasting_model' not in payload:
            raise ValueError('A field \'forecasting_model\' must be specified')
        self.forecasting = payload['forecasting_model']
        if type(self.forecasting) is not str:
            raise ValueError('The forecasting model must be a string')
        
    def _handle(self, *args, **kwargs):
        self.logger.debug("Start _handle")
        try:
            self._unpack(self.payload)
            self.logger.debug("done unpacking")
            out = logic.DetectorTrainer().train(
                                                label=self.identifier, 
                                                training = self.training,
                                                forecasting_model=self.forecasting, 
                                                )
            return {
                'code': 201,
                'result': out,
            }, 201
        except ValueError as e:
            return {
                'code': 400,
                'name': 'BadRequest',
                'description': str(e),
            }, 400

class ShowDetector(Controller):
    def __init__(self, identifier, version):
        self.label = identifier
        self.version = version

    def _handle(self, *args, **kwargs):
        try:
            env = logic.DetectorTrainer().retrieve_env(self.label, self.version)
            summ = logic.Summary()
            summ.load(os.path.join(env, constants.files.detector_summary))
            return {
                'code': 200,
                'summary': summ.values,
            }, 200
        except ValueError as e:
            return {
                'code': 400,
                'name': 'BadRequest',
                'description': str(e),
            }, 400

class ShowDetectorHistory(Controller):
    def __init__(self, identifier, version):
        self.label = identifier
        self.version = version

    def _handle(self, *args, **kwargs):
        try:
            env = logic.DetectorTrainer().retrieve_env(self.label, self.version)
            history = detector.History()
            history.load(os.path.join(env, constants.files.detector_history))
            return {
                'code': 200,
                'history': history.values,
            }, 200
        except ValueError as e:
            return {
                'code': 400,
                'name': 'BadRequest',
                'description': str(e),
            }, 400

class ShowDetectorParameters(Controller):
    def __init__(self, identifier, version):
        self.label = identifier
        self.version = version
    
    def _handle(self, *args, **kwargs):
        try:
            dt = logic.DetectorTrainer()
            dt.load_detector(self.label, self.version)
            params = {}
            if dt.loaded:
                params = dt._detector.params
            return {
                'code': 200,
                'params': params,
            }, 200
        except ValueError as e:
            return {
                'code': 400,
                'name': 'BadRequest',
                'description': str(e),
            }, 400

class Detect(Controller):
    def __init__(self, identifier, version, request):
        super().__init__()
        self.logger.info("{}-{}: Request: {}".format(identifier, version, request.get_json()))
        self.identifier = identifier
        self.version = version
        self.request = request
        self.payload = request.get_json()
        self.logger.debug("End __init__")
    
    def _unpack(self, payload):
        self.logger.debug("Unpacking: check data")
        self.data = payload.get('data', None)
        if 'data' not in payload:
            raise ValueError('A parameter \'data\' must be specified')

        self.logger.debug("Unpacking: check store")
        self.store = payload.get('store', False)
        if type(self.store) is not bool:
            raise ValueError('The \'store\' parameter must be a boolean')

        self.logger.debug("Unpacking: check pre_load")
        if 'pre_load' not in payload:
            self.pre_load = None
        else:
            self.pre_load = dict()
            if 'label' in payload['pre_load'] and \
                    'dataset' in payload['pre_load'] and \
                    'column' in payload['pre_load']:
                self.pre_load['label'] = payload['pre_load']['label']
                if type(self.pre_load['label']) is not str:
                    raise ValueError('The pre_load label must be a string')

                self.pre_load['dataset'] = payload['pre_load']['dataset']
                if type(self.pre_load['dataset']) is not str:
                    raise ValueError('The pre_load dataset must be a string')

                self.pre_load['column'] = payload['pre_load']['column']
                if type(self.pre_load['column']) is not str:
                    raise ValueError('The pre_load column must be a string')
        self.logger.debug("End Unpacking")


    def _handle(self, *args, **kwargs):
        try:
            self.logger.debug("Start _handle")
            self._unpack(self.payload)
            trainer = logic.DetectorTrainer()
            trainer.load_detector(self.identifier, self.version)
            self.logger.debug("Trainer ended loading model")
            if not trainer.loaded:
                return {
                    'code': 409,
                    'name': 'Conflict',
                    'description': 'The detector @{:s}.{:s} is currently not available'.format(self.identifier, version),
                }, 404
            else:
                out = trainer.detect(self.data, self.store, self.pre_load)
                self.logger.debug("Done detction")
                return {
                    'code': 200,
                    'results': out
                }, 200
        except ValueError as e:
            return {
                'code': 400,
                'name': 'BadRequest',
                'description': str(e),
            }, 400
        except Exception as e:
            self.logger.warning(str(e))
