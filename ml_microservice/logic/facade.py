import logging
from threading import Thread
import time

from ml_microservice.conversion import Xml2Csv
from ml_microservice.logic import metadata
from ml_microservice.logic import timeseries_lib
from ml_microservice.logic.timeseries_lib import TimeseriesLibrary
from ml_microservice.logic.detector_lib import AnomalyDetectorsLibrary

from ml_microservice.anomaly_detection.factory import Factory
from ml_microservice.anomaly_detection.loaders import Loader
from ml_microservice.anomaly_detection.preprocessing import Preprocessor
from ml_microservice.anomaly_detection.tuners import Tuner
from ml_microservice.anomaly_detection import evaluators
from ml_microservice.anomaly_detection.evaluators import Evaluator

class LogicFacade():
    def __init__(self):
        log_id = str(self.__class__.__name__).lower()
        self.logger = logging.getLogger(log_id)


    def convert_xml(self, id_patterns, ignore_patterns, store, 
                    xml, group, override):
        x2c = Xml2Csv(
            ignore_patterns = ignore_patterns,
            id_patterns = id_patterns
        )
        dfs = x2c.parse(xml)
        self.logger.debug("Parsings: {}".format(dfs))
        if store:
            ts_lib = TimeseriesLibrary()
            for dfID, df in dfs.items():
                ts_lib.save(group, dfID, df, override)
        return dfs


    def list_ts(self):
        return TimeseriesLibrary().timeseries


    def explore_ts_dim(self, groupID, dimID):
        dset_lib = TimeseriesLibrary()
        if not dset_lib.has_group(groupID):
            raise ValueError('Group \'{:s}\' not found'.format(groupID))
        if not dset_lib.has(groupID, dimID):
            raise ValueError('Dimension \'{:s}\' not found in group \'{:s}\''.format(dimID, groupID))
        df = dset_lib.fetch(groupID, dimID)
        return df


    def explore_ts(self, groupID, dimID, tsID):
        ts_lib = TimeseriesLibrary()
        if not ts_lib.has_group(groupID):
            raise ValueError("Group \'{:s}\' not found".format(groupID))
        if not ts_lib.has(groupID, dimID):
            raise ValueError("Dimension \'{:s}\' not found in group \'{:s}\'".format(groupID, dimID))
        df = ts_lib.fetch(groupID, dimID)
        if tsID not in df:
            raise ValueError("Timeseries ID \'{:s}\' not found in \'{:s}-{:s}\'".format(tsID, groupID, dimID))
        vs = df[tsID]
        return vs


    def list_methods(self):
        return Factory().get_types()


    def list_save_detectors(self):
        return AnomalyDetectorsLibrary().list()


    def detector_metadata(self, mID, version):
        ad_lib = AnomalyDetectorsLibrary()
        if not ad_lib.has(mID, version):
            raise RuntimeError("Model \'{:s}/{:s}\' not found")
        metadata = ad_lib.retrieve_metadata(mID, version)
        return metadata.to_dict()


    def detector_history(self, mID, version):
        ad_lib = AnomalyDetectorsLibrary()
        if not ad_lib.has(mID, version):
            raise RuntimeError("Model \'{:s}/{:s}\' not found".format(mID, version))
        env = ad_lib.retrieve_env(mID, version)
        return evaluators.load_history(env.root)


    def detector_parameters(self, mID, version):
        ad_lib = AnomalyDetectorsLibrary()
        if not ad_lib.has(mID, version):
            raise RuntimeError("Model \'{:s}/{:s}\' not found".format(mID, version))
        env = ad_lib.retrieve_env(mID, version)
        meta = metadata.load(env.root)
        loader = Factory().get_loader(meta.model_type)
        model = loader.load(env.assets)
        return model.get_params()


    def detector_train(self, mID, groupID, dimID, tsID, 
                        method):
        ts_lib = TimeseriesLibrary()
        if not ts_lib.has(groupID, dimID):
            raise ValueError("Timeseries (\'{:s}\', \'{:s}\') not found".format(groupID, dimID))
            
        if tsID == ts_lib.date_col:
            raise RuntimeError("The tsID \'{:s}\' is not a valid ID. Pick another".format(tsID))

        factory = Factory()
        if not factory.has(method):
            raise RuntimeError("The method \'{:s}\' is not supported. Pick another".format(method))
        ad_lib = AnomalyDetectorsLibrary()
        if ad_lib.has(mID):
            raise RuntimeError("ModelID \'{:s}\' already used".format(mID))

        v0, env, meta = ad_lib.create_env(mID)
        meta.set_type(method)
        meta.set_ts(groupID, dimID, tsID)
        ts = ts_lib.fetch_ts(groupID, dimID, tsID)
        tuner = factory.get_tuner(method)
        
        # Training
        t0 = time.time()
        train_time = 0
        preproc = Preprocessor(ts)
        tuner.tune(preproc)

        last_train_IDX = preproc.last_train_IDX_
        last_dev_IDX = getattr(preproc, "last_dev_IDX_", last_train_IDX)
        train_time = time.time() - t0
        
        model = tuner.best_model_
        model.save(env.assets)
        tuner.save_results(env.root)
        meta.set_training_info(
            last_train_IDX = last_train_IDX,
            last_dev_IDX = last_train_IDX,
            total_time = train_time,
            best_config = tuner.best_config_
        )
        meta.save(env.root)
        return dict(version=v0, train_time=train_time,
                        last_train_IDX=last_train_IDX,
                        last_dev_IDX=last_dev_IDX,
                        best_config=tuner.best_model_)


    def detector_predict(self, mID, version, values, dates):
        ad_lib = AnomalyDetectorsLibrary()
        if not ad_lib.has(mID, version):
            raise RuntimeError("Model \'{:s}/{:s}\' not found")
        
        env = ad_lib.retrieve_env(mID, version)
        meta = metadata.load(env.root)
        if not meta.is_training_done():
            raise ValueError('The anomaly detector @{:s}.{:s} is currently not available'.format(mID, version))
        loader = Factory().get_loader(meta.model_type)
        ts = timeseries_lib.compose_ts(values, dates)
        
        t0 = time.time()
        model = loader.load(env.assets)
        preproc = Preprocessor(ts)
        preproc.for_prediction()
        y_hat = model.predict(preproc.values)

        prediction_time = time.time() - t0
        self.logger.debug("Done detction")
        res = dict(
            y_hat=y_hat, 
            pred_time=prediction_time
        )
        if hasattr(model, "predict_prob_"):
            res["predict_prob"] = model.predict_prob_
        if hasattr(model, "forecast_"):
            res["forecast"] = model.forecast_
        return res
    

    def detector_eval(self, mID, version, forget):
        ad_lib = AnomalyDetectorsLibrary()
        if not ad_lib.has(mID, version):
            raise RuntimeError("Model \'{:s}/{:s}\' not found")
        
        env = ad_lib.retrieve_env(mID, version)
        meta = metadata.load(env.root)
        if not meta.is_training_done():
            raise ValueError('The anomaly detector @{:s}.{:s} is currently not available'.format(mID, version))
        
        factory = Factory()
        evaluator = factory.get_evaluator(meta.model_type)
        ts_lib = TimeseriesLibrary()
        
        groupID, dimID, tsID = meta.get_ts()
        ts = ts_lib.fetch_ts(groupID, dimID, tsID)

        t0 = time.time()
        evaluator.load_model(env.assets)

        preproc = Preprocessor(ts, start_from = meta.last_dev_IDX)
        scores = evaluator.evaluate(preproc, forget)

        if hasattr(evaluator, "forecast_"):
            self.forecast_ = getattr(evaluator, "forecast_")
        if hasattr(evaluator, "predict_prob_"):
            self.predict_prob_ = getattr(evaluator, "predict_prob_")
        eval_time = time.time() - t0
        y_hat = evaluator.prediction_

        if not forget and evaluator.is_performance_dropping():
            evaluator.save_results(env.root)
            t = Thread(
                target = self.detector_train, 
                args = (
                    mID,
                    groupID,
                    dimID,
                    tsID,
                    meta.model_type(),
                )
            )
            t.start()
        self.logger.debug("Done detction")

        resp = dict(
            values=preproc.values,
            y_hat = y_hat,
            eval_time = eval_time,
            scores = scores
        )
        if preproc.dates is not None:
            resp["dates"] = preproc.dates
        if hasattr(evaluator, "forecast_"):
            resp["forecast"] = evaluator.forecast_
        if hasattr(evaluator, "predict_prob_"):
            resp["predict_prob"] = evaluator.predict_prob_
        return resp