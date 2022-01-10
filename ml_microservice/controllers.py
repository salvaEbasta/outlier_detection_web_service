from typing import Dict, List
import logging

from ml_microservice.logic.facade import LogicFacade
from ml_microservice import configuration as cfg

class Controller():
    def __init__(self, lvl = logging.DEBUG):
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
        """
        payload: {
            "xml": str,
            \["id_patterns": [str, ]],
            \["ignore_patterns": [str, ]],
            \["store": {
                "group": str,
                "override": bool
            }]
        }
        """
        self.logger.info("Unpack: {}".format(payload))

        # Mandatory
        if 'xml' not in payload:
            raise ValueError('The field \'xml\' must be specified')
        self.xml = payload['xml']

        # Optionals
        if 'id_patterns' in payload and type(payload['id_patterns']) is not str:
            raise ValueError('The field \'id_patterns\' must be a list of strings')
        self.id_patterns = [payload.get('id_patterns')] if 'id_patterns' in payload else []
        
        if 'ignore_patterns' in payload and type(payload['ignore_patterns']) is not List[str]:
            raise ValueError('The field \'ignore_patterns\' must be a list of strings')
        self.ignore_patterns = payload.get('ignore_patterns', [])

        self.store = False
        if 'store' in payload and type(payload['store']) is not dict:
            raise ValueError('The field \'store\' must be a dict if specified')
        self.store_info = payload.get('store', {})
        if len(self.store_info):
            self.store = True
            if 'group' not in self.store_info or type(self.store_info["group"]) is not str:
                raise ValueError('The field \'group\' must be specified if the series is to be stored')
            if "override" not in self.store_info or type(self.store_info["override"]) is not bool:
                raise ValueError('The field \'override\' must be specified if the series is to be stored')
        self.group = self.store_info.get("group", None)
        self.override = self.store_info.get("override", False)

    def _handle(self, *args, **kwargs):
        """
        {
            "code": 200,
            "extracted": [{
                "dimension": str,
                "data": [stuff,]
            }, ],

        }
        """
        try:
            self._unpack(self.payload)
            dfs = LogicFacade().convert_xml(
                id_patterns=self.id_patterns,
                ignore_patterns=self.ignore_patterns,
                store=self.store,
                xml=self.xml,
                override=self.override,
                groupID=self.group
            )
            dC = cfg.timeseries.date_column
            tmp = {}
            for dfID, df in dfs.items():
                if dC in df.columns:
                    df[dC] = df[dC].astype("string")
                tmp[dfID] = {c: df[c].fillna(cfg.timeseries.nan_str).to_list() 
                                for c in df.columns}
            resp = {
                'code': 200,
                'extracted': [{
                    'dimension': dfID,
                    'data': df_dict,
                    } for dfID, df_dict in tmp.items()
                ],
            }
            if self.store:
                resp.update({
                        "stored": {
                            "group": self.group,
                            "override": self.override
                        }
                    })
            return resp, 200
        except ValueError as e:
            return {
                'code': 400,
                'name': 'BadeRequest',
                'description': str(e),
            }, 400

class ListTimeseries(Controller):
    def __init__(self):
        super().__init__()

    def _handle(self, *args, **kwargs):
        """
        {
            "code": int,
            "timeseries": [{
                "group": str,
                "dimensions": [str, ]
            }, ],
        }
        """
        return {
            'code': 200,
            'timeseries': LogicFacade.list_ts(),
        }, 200

class ExploreTSDim(Controller):
    def __init__(self, group, dimension):
        super().__init__()
        self.group = group
        self.dimension = dimension

    def _handle(self, *args, **kwargs):
        """
        {
            "code": int,
            "group": {
                "id": str,
                "dimension": {
                    "id": str,
                    "tsIDs": [str, ],
                    "shape": [int, ]
                },
            },
        }
        """
        try:
            df = LogicFacade().explore_ts_dim(
                self.group,
                self.dimension
            )
            df = df.drop(cfg.timeseries.date_column, axis = 1)
            return {
                'code': 200,
                'group': {
                    "id": self.group,
                    "dimension": {
                        "id": self.dimension,
                        "tsIDs": list(df.columns),
                        "shape": [len(df), len(df.columns), ]
                    },
                },
            }, 200
        except ValueError as ve:
            return {
                'code': 404,
                'error': 'NotFound',
                'description': str(ve),
            }, 404

class ExploreTS(Controller):
    def __init__(self, group, dimension, tsID):
        super().__init__()
        self.group = group
        self.dimension = dimension
        self.tsID = tsID

    def _handle(self, *args, **kwargs):
        """
        {
            "code": int,
            "tsID": str,
            "values": {idx: val, }
        }
        """
        try:
            vs = LogicFacade().explore_ts(
                self.group,
                self.dimension,
                self.tsID
            )
            if self.tsID == cfg.timeseries.date_column:
                vs = vs.astype("string")
            return {
                'code': 200,
                'tsID': self.tsID,
                'values': vs.fillna(cfg.timeseries.nan_str).to_dict()
            }, 200
        except ValueError as ve:
            return {
                'code': 404,
                'name': 'NotFound',
                'description': str(ve),
            }, 404

class ListMethods(Controller):
    def __init__(self):
        super().__init__()

    def _handle(self, *args, **kwargs):
        """
        {
            "code": int,
            "methods": [str, ]
        }
        """
        return {
            'code': 200,
            'methods': LogicFacade().list_methods(),
        }, 200

class ListSavedDetectors(Controller):
    def __init__(self):
        super().__init__()

    def _handle(self, *args, **kwargs):
        """
        {
            "code": int,
            "saved": [{
                "mID": str, 
                "versions": [str, ]
            }, ]
        }
        """
        return {
            'code': 200,
            'saved': LogicFacade().list_save_detectors(),
        }, 200

class DetectorTrain(Controller):
    def __init__(self, request):
        super().__init__()
        self.logger.info(f"Request: {request.get_json()}")
        self.request = request
        self.payload = request.get_json()
        self.logger.debug("End __init__")

    def _unpack(self, payload):
        """
        {
            "train": {
                "groupID": str,
                "dimID": str,
                "tsID": str,
            },
            "method": str,
            "mID": str,
        }
        """
        self.type = None
        
        # Mandatory
        self.logger.debug('Payload: check \'mID\'')
        if 'mID' not in payload:
            raise ValueError('The field \'mID\' must be specified')
        if type(payload["mID"]) is not str:
            raise ValueError('The mID must be a string')
        self.mID = payload['mID']
        
        self.logger.debug('Payload: check \'train\'')
        if "train" not in payload:
            raise ValueError('A field \'train\' must be specified. Must be a dict with keys \'groupID\', \'dimID\', \'tsID\'')
        if 'groupID' not in payload['train'] \
            or "dimID" not in payload['train'] \
            or "tsID" not in payload['train']:
            raise ValueError('The field \'train\' must be a dict with keys \'groupID\', \'dimID\', \'tsID\'')
        
        if type(payload["train"]["groupID"]) is not str:
            raise ValueError('The field \'train\'-\'groupID\' must be a string')
        self.groupID = payload["train"]["groupID"]
        
        if type(payload["train"]["dimID"]) is not str:
            raise ValueError('The field \'train\'-\'dimID\' must be a string')
        self.dimID = payload["train"]["dimID"]

        if type(payload["train"]["tsID"]) is not str:
            raise ValueError('The field \'train\'-\'tsID\' must be a string')
        self.tsID = payload["train"]["tsID"]
        
        if "method" not in payload or type(payload["method"]) is not str:
            raise ValueError('The field \'method\' must be specified and have a string value')
        self.method = payload["method"]
        
    def _handle(self, *args, **kwargs):
        """
        {
            "code": int,
            "train": {
                "timeseries": {
                    "groupID": str,
                    "dimID": str,
                    "tsID": str,
                },
                "overview": {
                    "total_time(s)": float,
                    "best_config": dict,
                    "last_train_IDX": int,
                    "last_dev_IDX": int,
                }
            },
            "method": {
                "id": str,
            },
            "model": {
                "mID": str,
                "version": str
            }
        }
        """
        self.logger.debug("Start _handle")
        try:
            self._unpack(self.payload)
            self.logger.debug("done unpacking")
            result = LogicFacade().detector_train(
                mID=self.mID,
                groupID=self.groupID,
                dimID=self.dimID,
                tsID=self.tsID,
                method=self.method,
            )
            return {
                "code": 201,
                "model": {
                    "mID": self.mID,
                    "version": result["version"],
                },
                "train": {
                    "timeseries": {
                        "groupID": self.groupID,
                        "dimID": self.dimID,
                        "tsID": self.tsID,
                    },
                    "overview": {
                        "total_time(s)": result["train_time"],
                        "last_train_IDX": result["last_train_IDX"],
                        "last_dev_IDX": result["last_dev_IDX"],
                        "best_config": result["best_config"]
                    }
                },
                "method": {
                    "id": self.method,
                },
            }, 201
        except ValueError as e:
            return {
                "code": 404,
                "name": "NotFound",
                "description": str(e),
            }, 404
        except RuntimeError as e:
            return {
                'code': 400,
                'name': 'BadRequest',
                'description': str(e),
            }, 400

class DetectorMetadata(Controller):
    def __init__(self, mID, version):
        self.mID = mID
        self.version = version

    def _handle(self, *args, **kwargs):
        """
        {
            "code": int,
            "metadata": {
                "status: {
                    "name": str,
                    "code": int,
                },
                "type": str,
                "created": str,
                "timeseries": {
                    \["groupID": str,
                      "dimID": str,
                      "tsID": str
                    ]
                },
                "training": {
                    \["total_time(s)": float,
                      "last_train_IDX": int,
                      \["last_dev_IDX": int,]
                    ]
                }
            }
        }
        """
        try:
            return {
                'code': 200,
                'metadata': LogicFacade().detector_metadata(self.mID, self.version),
            }, 200
        except ValueError as e:
            return {
                'code': 400,
                'name': 'BadRequest',
                'description': str(e),
            }, 400
        except RuntimeError as e:
            return {
                "code": 404,
                "name": "NotFound",
                "description": str(e)
            }, 404

class ShowDetectorHistory(Controller):
    def __init__(self, mID, version):
        self.mID = mID
        self.version = version

    def _handle(self, *args, **kwargs):
        """
        {
            "code": int,
            "mID": str,
            "version": str,
            "history": {
                str: [,]
            }
        }
        """
        try:
            history = LogicFacade().detector_history(self.mID, self.version)
            for c in history.columns:
                history[c] = history[c].fillna(cfg.timeseries.nan_str)
                if c == cfg.evaluator.date_column:
                    history[c] = history[c].astype("string")
            return {
                'code': 200,
                "mID": self.mID,
                "version": self.version,
                'history': {c: history[c].to_list() 
                                for c in history.columns}
            }, 200
        except ValueError as e:
            return {
                'code': 400,
                'name': 'BadRequest',
                'description': str(e),
            }, 400
        except RuntimeError as e:
            return {
                "code": 404,
                "name": "NotFound",
                "description": str(e)
            }, 404

class DetectorParameters(Controller):
    def __init__(self, mID, version):
        self.mID = mID
        self.version = version
    
    def _handle(self, *args, **kwargs):
        """
        {
            "code": int,
            "mID": str,
            "version": str,
            "params": dict,
        }
        """
        try:
            
            return {
                'code': 200,
                "mID": self.mID,
                "version": self.version,
                'params': LogicFacade().detector_parameters(self.mID, self.version),
            }, 200
        except ValueError as e:
            return {
                'code': 400,
                'name': 'BadRequest',
                'description': str(e),
            }, 400
        except RuntimeError as e:
            return {
                "code": 404,
                "name": "NotFound",
                "description": str(e)
            }, 404

class DetectPredict(Controller):
    def __init__(self, mID, version, request):
        super().__init__()
        self.logger.info("{}-{}: Request: {}".format(mID, version, request.get_json()))
        self.mID = mID
        self.version = version
        self.request = request
        self.payload = request.get_json()
        self.logger.debug("End __init__")
    
    def _unpack(self, payload):
        """
        {
            "data": {
                \["dates": [,],]
                "values": [,],
            }
        }
        """
        self.logger.debug("Unpacking: check data")
        self.data = payload.get('data', None)
        if self.data is None:
            raise ValueError('A field \'data\' must be specified')
        if type(self.data) is not Dict:
            raise ValueError('The field \'data\' must be a dict')
        
        self.values = self.data.get("values", None)
        if "values" is None:
            raise ValueError('The field \'values\' must be specified under \'data\'')
        if type(self.values) is not List[float]:
            raise ValueError('The field \'data\':\'values\' must be a list of floats')
        
        self.dates = self.data.get("dates", None)
        if self.dates is not None:
            if type(self.dates) is not List[str]:
                raise ValueError('The field \'data\':\'dates\' must be a list of strings')
            if len(self.dates) != len(self.values) :
                raise ValueError('The field \'data\':\'dates\' must be same length of \'data\':\'values\'')
        self.logger.debug("End Unpacking")

    def _handle(self, *args, **kwargs):
        """
        {
            "code": int,
            "mID": str,
            "version": str,
            "data": {
                "values": [,],
                \["dates": [,],]
            }
            "predictions": {
                "anomaly_class": [,],
                \["anomaly_score": [,],]
                \["forecast": [,],]
                "total_time(s)": float,
            },
        }
        """
        try:
            self.logger.debug("Start _handle")
            self._unpack(self.payload)
            tmp = LogicFacade().detector_predict(
                mID=self.mID,
                version=self.version,
                values=self.values,
                dates=self.dates
            )
            resp = {
                'code': 200,
                "mID": self.mID,
                "version": self.version,
                "data": self.data,
                'predictions': {
                    "anomaly_class": tmp["y_hat"],
                    "total_time(s)": tmp["pred_time"],
                }
            }
            if "predict_prob" in tmp:
                resp["predictions"]["anomaly_score"] = tmp["predict_prob"]
            if "forecast" in tmp:
                resp["predictions"]["forecast"] = tmp["forecast"]
            return resp, 200
        except ValueError as e:
            return {
                'code': 400,
                'name': 'BadRequest',
                'description': str(e),
            }, 400
        except RuntimeError as e:
            return {
                "code": 404,
                "name": "NotFound",
                "description": str(e)
            }, 404
        except Exception as e:
            self.logger.warning(str(e))

class DetectorEvaluate(Controller):
    def __init__(self, mID, version):
        super().__init__()
        self.logger.info("{}-{}".format(mID, version))
        self.mID = mID
        self.version = version
        self.payload = {"forget": False,}
        self.logger.debug("End __init__")
    
    def set_request(self, r):
        self.request = r
        self.payload = r.get_json()
        return self

    def _unpack(self, payload):
        """
        {
            "forget": bool,
        }
        """
        self.logger.debug("Unpacking: check data")
        if "forget" not in payload:
            raise ValueError('The field \'forget\' must be specified')
        self.forget = payload.get("forget", True)
        if type(self.forget) is not bool:
            raise ValueError('The field \'forget\' must be a bool')
        self.logger.debug("End Unpacking")

    def _handle(self, *args, **kwargs):
        """
        {
            "code": int,
            "mID": str,
            "version": str,
            "data": {
                "values": [,],
                \["dates": [,],]
            }
            "evaluation": {
                "anomaly_class": [,],
                \["anomaly_score": [,],]
                \["forecast": [,],]
                "total_time(s)": float,
            },
        }
        """
        try:
            tmp = LogicFacade().detector_eval(self.mID, self.version)
            resp = {
                'code': 200,
                "mID": self.mID,
                "version": self.version,
                "data": {
                    "values": tmp["values"],
                },
                'evaluation': {
                    "anomaly_class": tmp["y_hat"],
                    "total_time(s)": tmp["eval_time"],
                    "scores": tmp["scores"],
                }
            }
            if "dates" in tmp:
                resp["data"]["dates"] = tmp["dates"]
            if "predict_prob" in tmp:
                resp["predictions"]["anomaly_score"] = tmp["predict_prob"]
            if "forecast" in tmp:
                resp["predictions"]["forecast"] = tmp["forecast"]
            return resp, 200
        except ValueError as e:
            return {
                'code': 400,
                'name': 'BadRequest',
                'description': str(e),
            }, 400
        except RuntimeError as e:
            return {
                "code": 404,
                "name": "NotFound",
                "description": str(e)
            }, 404
        except Exception as e:
            self.logger.warning(str(e))
